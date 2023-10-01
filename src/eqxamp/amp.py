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


def cast_if_possible(to_type: Optional[Type], x: Any) -> Any:
    if to_type is None:
        return x
    if not eqx.is_array(x):
        return x
    if jnp.can_cast(x, to_type, "unsafe"):
        return x.astype(to_type)
    return x


def get_type(x: Any) -> Optional[Type]:
    if not eqx.is_array(x):
        return None
    return x.dtype


def cast_tree(to_type: Optional[Type], tree: PyTree) -> PyTree:
    if to_type is None:
        return tree
    cast_leaf = eqx.Partial(cast_if_possible, to_type)
    return jtu.tree_map(cast_leaf, tree)


def get_children(tree: PyTree) -> List[Any]:
    return jtu.tree_flatten(tree, lambda node: node is not tree)


def is_child_fn(tree: PyTree):
    return lambda node: node is not tree


def cast_module(module: eqx.Module, cast_child: Callable[PyTree, PyTree]) -> eqx.Module:
    return jtu.tree_map(cast_child, module, is_leaf=is_child_fn(module))


def with_mixed_precision(
    orig_module: eqx.Module,
    output_type: Optional[Type] = None,
    compute_type: Optional[Type] = None,
    storage_type: Optional[Type] = None,
    recurse_fn: Optional[Callable[eqx.Module, eqx.Module]] = None,
) -> eqx.Module:
    def cast_child(child: PyTree) -> PyTree:
        if isinstance(child, MixedPrecisionWrapper):
            raise TypeError("Attempting to apply mixed precision to a module that already has mixed precision applied!")

        if isinstance(child, eqx.Module):
            if recurse_fn:
                return recurse_fn(child)
            else:
                return child

        if callable(child):
            return MixedPrecisionWrapper(child, output_type, storage_type, compute_type)
        return cast_tree(storage_type, child)

    casted_module = cast_module(orig_module, cast_child)

    return eqx.module_update_wrapper(
        MixedPrecisionWrapper(casted_module, output_type, storage_type, compute_type), casted_module
    )


def cast_fn(fn: Callable, output_type: Optional[Type] = None, compute_type: Optional[Type] = None):
    @jtu.Partial
    @wraps(fn)
    def mixed_fn(*args, **kwargs):
        args = cast_tree(compute_type, args)
        kwargs = cast_tree(compute_type, kwargs)
        output = fn(*args, **kwargs)
        result = cast_tree(output_type, output)
        return result

    return mixed_fn


class MixedPrecisionWrapper(eqx.Module):
    _original_module: eqx.Module
    output_type: Optional[Type] = eqx.field(default=None, static=True)
    storage_type: Optional[Type] = eqx.field(default=None, static=True)
    compute_type: Optional[Type] = eqx.field(default=None, static=True)

    def __call__(self, *args, **kwargs) -> Any:
        def cast_child(child: PyTree) -> PyTree:
            if isinstance(child, eqx.Module):
                return child

            if callable(child):
                return MixedPrecisionWrapper(child, self.output_type, self.storage_type, self.compute_type)
            return cast_tree(self.compute_type, child)

        module = cast_module(self._original_module, cast_child)
        args = cast_tree(self.compute_type, args)
        kwargs = cast_tree(self.compute_type, kwargs)
        output = module(*args, **kwargs)
        result = cast_tree(self.output_type, output)
        return result

    @property
    def __wrapped__(self):
        return self._original_module


@dataclass
class MixedTypes:
    output_type: Optional[Type] = None
    compute_type: Optional[Type] = None
    storage_type: Optional[Type] = None


full_precision = MixedTypes(jnp.float32, jnp.float32, jnp.float32)
double_precision = MixedTypes(jnp.float64, jnp.float64, jnp.float64)


def default_amp_types(module: eqx.Module, parents: List[eqx.Module]) -> Optional[MixedTypes]:
    type_overrides = None
    if hasattr(module, 'amp_types'):
        return module.amp_types
    for module_class in AMP_OVERRIDES:
        if isinstance(module, module_class):
            type_overrides = AMP_OVERRIDES[module_class]
            break
    return type_overrides


def amp(
    orig_module: eqx.Module,
    types: MixedTypes = MixedTypes(jnp.float32, ml_dtypes.bfloat16, jnp.float32),
    type_override_fn: Callable[[eqx.Module, List[eqx.Module]], Optional[MixedTypes]] = default_amp_types,
    parents=[],
) -> eqx.Module:
    # recurse_fn will ignore provided types in order to avoid propogating overrides.
    # we replace output_type with compute_type since internal nodes should keep data as
    # compute type.
    recurse_fn = lambda m: amp(m, types, type_override_fn, parents=parents + [orig_module])
    mp_types = type_override_fn(orig_module, parents) or types
    return with_mixed_precision(orig_module, recurse_fn=recurse_fn, **asdict(mp_types))


class M(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    nm: eqx.nn.LayerNorm
    p: jax.Array

    def __init__(self, n: int, key: jrandom.PRNGKey):
        keys = jrandom.split(key, 3)
        self.l1 = eqx.nn.Linear(1, n, key=keys[0])
        self.nm = nn.LayerNorm(n)
        self.l2 = eqx.nn.Linear(n, 1, key=keys[2])
        self.p = jnp.ones(4)

    def __call__(self, x):
        return self.l2(self.nm(self.l1(x)))


AMP_OVERRIDES = {
    nn.LayerNorm: full_precision,
    nn.GroupNorm: full_precision,
    nn.SpectralNorm: full_precision,
    nn.BatchNorm: full_precision,
}


def all_finite(tree: PyTree) -> jax.Array:
    # from JMPi
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    finite = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return jnp.stack(list(finite)).all()


class DynamicScalarState(NamedTuple):
    patience: jax.Array = jnp.array(2000)
    adjust_factor: jax.Array = jnp.array(2.0)
    scalar: jax.Array = jnp.array(2**15, dtype=jnp.float32)
    count: jax.Array = jnp.array(0)


def increment_state(state):
    new_state = jax.lax.cond(
        state.count >= state.patience,
        lambda state: DynamicScalarState(state.patience, state.adjust_factor, state.scalar * state.adjust_factor, 0.0),
        lambda state: DynamicScalarState(state.patience, state.adjust_factor, state.scalar, state.count + 1.0),
        state,
    )

    return new_state


def decrease_scalar(state):
    return DynamicScalarState(state.patience, state.adjust_factor, state.scalar / state.adjust_factor, 0.0)


def dynamic_scale_tx(transform: Callable[Any, Any], redo_on_nan: int = 0):
    def scaled_transform(fun, *args, **kwargs):
        @wraps(fun)
        def scaled_fun(*f_args, dynamic_scalar_state: DynamicScalarState, **f_kwargs):
            result = fun(*f_args, **f_kwargs)
            return result * dynamic_scalar_state.scalar.astype(result.dtype)

        transformed_fn = transform(scaled_fun, *args, **kwargs)

        def maybe_adjust_scalar(*f_args, dynamic_scalar_state: DynamicScalarState, **f_kwargs):
            # avoid type mismatch complaints later in case state is initialized
            # with raw python ints/floats
            state = jtu.tree_map(lambda x: jnp.array(x).astype(jnp.float32), dynamic_scalar_state)

            results = transformed_fn(*f_args, dynamic_scalar_state=state, **f_kwargs)

            results = jtu.tree_map(lambda x: x.astype(state.scalar.dtype) / state.scalar, results)

            new_state = jax.lax.cond(all_finite(results), increment_state, decrease_scalar, state)

            return results, new_state

        if redo_on_nan == 0:
            return maybe_adjust_scalar

        def adjust_scalar_until_finite(*f_args, dynamic_scalar_state: DynamicScalarState, **f_kwargs):
            redo_count = jnp.array(0, jnp.int32)

            results, new_state = maybe_adjust_scalar(*f_args, dynamic_scalar_state=dynamic_scalar_state, **f_kwargs)

            init_val = (results, new_state, redo_count)

            def cond_fun(results__state__redo_count):
                results, state, redo_count = results__state__redo_count
                return jnp.logical_and(jnp.logical_not(all_finite(results)), redo_count < redo_on_nan)

            def body_fun(results__state__redo_count):
                results, state, redo_count = results__state__redo_count
                results, state = maybe_adjust_scalar(*f_args, dynamic_scalar_state=state, **f_kwargs)
                return (results, state, redo_count + 1)

            results, new_state, redo_count = jax.lax.while_loop(
                cond_fun=cond_fun,
                body_fun=body_fun,
                init_val=init_val,
            )
            return results, new_state

        return adjust_scalar_until_finite

    return scaled_transform


if __name__ == '__main__':

    def cast_x(x):
        return x.astype(jnp.float16)

    x = jnp.array(2.0, dtype=jnp.float32)

    closed_jaxpr = jax.make_jaxpr(cast_x)(x)
    print("jaxpr: ", closed_jaxpr)

    def func(x):
        return x**2

    x = jnp.array(2.0, dtype=jnp.float16)
    scalar = DynamicScalarState(scalar=2**15)
    grad_fn = jax.jit(dynamic_scale_tx(jax.grad, redo_on_nan=100)(func))
    for i in range(10):
        g, scalar = grad_fn(x, dynamic_scalar_state=scalar)
        print("i: ", i)
        print("scalar: ", scalar.scalar)
        print("g: ", g)
        print("dtype: ", g.dtype)

    m = M(3, jrandom.PRNGKey(0))
    # l = eqx.nn.Linear(10,10, key=jrandom.PRNGKey(0))
    wrapped = amp(m, MixedTypes(compute_type=ml_dtypes.float8_e4m3fnuz, storage_type=ml_dtypes.float8_e4m3fnuz))
    # wrapped = with_mixed_precision(
    #     m, compute_type=ml_dtypes.float8_e4m3fnuz, storage_type=ml_dtypes.bfloat16, recurse_fn=with_mixed_precision
    # )  # float8_e4m3fnuz)
    # eqx.module_update_wrapper(CastWrapper(m), m)
    a = jnp.ones(1)
    print(m(a))
    print(wrapped(a))
    # print(wrapped)
    # print(m)

    # print(jtu.tree_leaves_with_path(
    #       [
    #           [1,2],
    #           [3,
    #               [4,
    #                   [5]
    #               ]
    #           ]
    #       ]))

    # , is_leaf=lambda x: x is not m))
    # isinstance(x, eqx.Module))
