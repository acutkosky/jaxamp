import equinox as eqx
from equinox import nn
from typing import Callable, Type, Tuple, Any, Optional, List, NamedTuple
import ml_dtypes
from jax import random as jrandom
from jax import numpy as jnp
import jax
from jax import tree_util as jtu
from jaxtyping import Array, PyTree
from dataclasses import dataclass, astuple, asdict
from functools import wraps
from jax import core
from jax import lax
from jax._src.util import safe_map


def all_finite(tree: PyTree) -> jax.Array:
    # from JMP
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    finite = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return jnp.stack(list(finite)).all()


class DynamicScalerState(NamedTuple):
    patience: jax.Array = jnp.array(
        2000
    )  # number of non-inf/NaN iterations to wait before increasing the scaler
    adjust_factor: jax.Array = jnp.array(
        2.0
    )  # When increasing or decreasing the scaler, multiply or divide by this factor.
    scaler: jax.Array = jnp.array(2**15, dtype=jnp.float32)  # current scaler value
    count: jax.Array = jnp.array(
        0
    )  # number of non-inf/NaN iterations since the scaler was last increased.


def increment_state(state: DynamicScalerState) -> DynamicScalerState:
    new_state = jax.lax.cond(
        state.count >= state.patience,
        lambda state: DynamicScalerState(
            state.patience, state.adjust_factor, state.scaler * state.adjust_factor, 0.0
        ),
        lambda state: DynamicScalerState(
            state.patience, state.adjust_factor, state.scaler, state.count + 1.0
        ),
        state,
    )

    return new_state


def decrease_scaler(state: DynamicScalerState) -> DynamicScalerState:
    return DynamicScalerState(
        state.patience, state.adjust_factor, state.scaler / state.adjust_factor, 0.0
    )


def default_unscale_fn(results: Any, state: DynamicScalerState) -> Any:
    return jtu.tree_map(lambda x: x.astype(state.scaler.dtype) / state.scaler, results)


def value_and_grad_aux_unscale_fn(results: Any, state: DynamicScalerState) -> Any:
    (value, aux), grad = results
    value = default_unscale_fn(value, state)
    grad = default_unscale_fn(grad, state)
    return (value, aux), grad


def grad_aux_unscale_fn(results: Any, state: DynamicScalerState) -> Any:
    grad, aux = results
    grad = default_unscale_fn(grad, state)
    return grad, aux


def default_scale_fn(result: Any, state: DynamicScalerState) -> Any:
    if not eqx.is_array_like(result):
        value, aux = result
        value = state.scaler.astype(value.dtype) * value
        result = (value, aux)
    else:
        result = result * state.scaler.astype(result.dtype)
    return result


def dynamic_scale_tx(
    transform: Callable[Any, Any],
    redo_on_nan: int = 0,
    unscale_fn: Callable = default_unscale_fn,
    scale_fn: Callable = default_scale_fn,
):
    def scaled_transform(fun, *args, **kwargs):
        @wraps(fun)
        def scaled_fun(*f_args, _dynamic_scaler_state: DynamicScalerState, **f_kwargs):
            result = fun(*f_args, **f_kwargs)
            return scale_fn(result, _dynamic_scaler_state)

        transformed_fn = transform(scaled_fun, *args, **kwargs)

        def maybe_adjust_scaler(
            *f_args, dynamic_scaler_state: DynamicScalerState, **f_kwargs
        ):
            # avoid type mismatch complaints later in case state is initialized
            # with raw python ints/floats
            state = jtu.tree_map(
                lambda x: jnp.array(x).astype(jnp.float32), dynamic_scaler_state
            )

            results = transformed_fn(*f_args, _dynamic_scaler_state=state, **f_kwargs)

            results = unscale_fn(results, state)

            new_state = jax.lax.cond(
                all_finite(results), increment_state, decrease_scaler, state
            )

            return new_state, results

        if redo_on_nan == 0:
            return maybe_adjust_scaler

        def adjust_scaler_until_finite(
            *f_args, dynamic_scaler_state: DynamicScalerState, **f_kwargs
        ):
            redo_count = jnp.array(0, jnp.int32)

            new_state, results = maybe_adjust_scaler(
                *f_args, dynamic_scaler_state=dynamic_scaler_state, **f_kwargs
            )

            init_val = (new_state, results, redo_count)

            def cond_fun(state__results__redo_count):
                state, results, redo_count = state__results__redo_count
                return jnp.logical_and(
                    jnp.logical_not(all_finite(results)), redo_count < redo_on_nan
                )

            def body_fun(state__results__redo_count):
                state, results, redo_count = state__results__redo_count
                state, results = maybe_adjust_scaler(
                    *f_args, dynamic_scaler_state=state, **f_kwargs
                )
                return (state, results, redo_count + 1)

            new_state, results, redo_count = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=init_val,
            )
            return new_state, results

        return adjust_scaler_until_finite

    return scaled_transform


def dynamic_scale_grad(
    fun: Callable,
    *,
    has_aux: bool = False,
    redo_on_nan: int = 0,
    filter=True,
    **kwargs
):
    '''
    apply dynamic scaler to the grad function.

    Args:
        fun: function to differentiate
        has_aux: same meaning as in jax.grad
        redo_on_nan: if the output is nan, we will decrease the scaler
            and recompute this many times. If the output remains nan, give up
            and return it.
        filter: if True, differentiate with equinox.filter_grad, otherwise use jax.grad

    Returns:
        grad_fn: a function that behaves like the output of jax.grad except:
            1. has an extra required keyword argument dynamic_scaler_state
            2. the return value is now a tuple (next_dynamic_scaler_state, grads)
                of (next_dynamic_scaler_state, (grads, aux)) if has_aux=True
    '''
    if has_aux:
        unscale_fn = grad_aux_unscale_fn
    else:
        unscale_fn = default_unscale_fn

    if filter:
        tx = eqx.filter_grad
    else:
        tx = jax.grad

    grad_fn = dynamic_scale_tx(tx, redo_on_nan=redo_on_nan, unscale_fn=unscale_fn)(
        fun, has_aux=has_aux, **kwargs
    )
    return grad_fn


def dynamic_scale_value_and_grad(
    fun: Callable,
    *,
    has_aux: bool = False,
    redo_on_nan: bool = 10,
    filter=True,
    **kwargs
):
    '''
    apply dynamic scaler to the value_and_grad function.

    Args:
        fun: function to differentiate
        has_aux: same meaning as in jax.grad
        redo_on_nan: if the output is nan, we will decrease the scaler
            and recompute this many times. If the output remains nan, give up
            and return it.
        filter: if True, differentiate with equinox.filter_value_and_grad, otherwise use jax.value_and_grad

    Returns:
        grad_fn: a function that behaves like the output of jax.value_and_grad except:
            1. has an extra required keyword argument dynamic_scaler_state
            2. the return value is now a tuple (next_dynamic_scaler_state, (value, grads))
                of (next_dynamic_scaler_state, ((value, aux), grads)) if has_aux=True
    '''
    if has_aux:
        unscale_fn = value_and_grad_aux_unscale_fn
    else:
        unscale_fn = default_unscale_fn
    if filter:
        tx = eqx.filter_value_and_grad
    else:
        tx = jax.value_and_grad

    grad_fn = dynamic_scale_tx(tx, redo_on_nan=redo_on_nan, unscale_fn=unscale_fn)(
        fun, has_aux=has_aux, **kwargs
    )
    return grad_fn
