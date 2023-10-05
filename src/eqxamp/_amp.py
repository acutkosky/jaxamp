import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
from functools import wraps

from jax import core
from jax import lax
from jax._src.util import (
    safe_zip,
    safe_map,
    curry,
    tuple_insert,
    tuple_delete,
    as_hashable_function,
    HashableFunction,
    HashableWrapper,
    weakref_lru_cache,
    partition_list,
)
from jax._src.core import Jaxpr, Atom, Literal, Var, last_used, clean_up_dead_vars, eval_jaxpr
from jax._src import source_info_util
from typing import Any, Callable, ClassVar, DefaultDict, Generic, NamedTuple, TypeVar, Union, cast, overload
from types import MappingProxyType
from collections import defaultdict
import ml_dtypes
import equinox as eqx
from jax import tree_util as jtu
from contextlib import contextmanager

zip, unsafe_zip = safe_zip, zip
map, unsafe_map = safe_map, map


def use_original_precision(compute_dtype, *invars, **bind_params):
    return invars, bind_params


@curry
def cast_if_float(dtype, value):
    if eqx.is_inexact_array(value):
        return value.astype(dtype)
    return value


def use_low_precision(compute_dtype, *invars, **bind_params):
    invars = map(cast_if_float(compute_dtype), invars)
    bind_params = dict(bind_params)
    if "preferred_element_type" in bind_params:
        bind_params["preferred_element_type"] = compute_dtype
    return invars, bind_params


low_precision_primitives = [
    lax.dot_general_p,
    lax.add_p,
    lax.sub_p,
]

# this dict specifies how different ops should be treated
# for AMP. Right now it is very simple: we move to low
# precision for additions, subtractions and
# tensor contractions  (e.g. matrix multiplies, convs etc).
# Otherwise we do not do anything.
default_amp_policy = {op: use_low_precision for op in low_precision_primitives} | {
    "amp_default": use_original_precision,
    "amp_stop": use_original_precision,
}


def amp_stop(f=None):
    context = jax.named_scope("amp_stop")
    if f is None:
        return context
    else:
        return context(f)


def amp_eval_jaxpr(compute_dtype, amp_policy, closed_jaxpr: Jaxpr, *args, propagate_source_info=True):
    jaxpr = closed_jaxpr.jaxpr

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        env[v] = val

    env: dict[Var, Any] = {}
    map(write, jaxpr.constvars, closed_jaxpr.consts)
    map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = map(read, eqn.invars)
            scopes = str(name_stack).split("/")  # [elem.name for elem in name_stack]
            scopes.append(eqn.primitive)
            scopes.append("amp_default")
            print("scopes: ",scopes)
            print("policy keys: ",list(amp_policy.keys()))
            for scope in scopes:
                if scope in amp_policy:
                    invars, bind_params = amp_policy[scope](compute_dtype, *invars, **bind_params)
                    break
            outvar_dtypes = map(lambda x: x.aval.dtype, eqn.outvars)
            ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
            # We will trust in the compiler to eliminate noop casts to full precision
            # followed by casts back down.
            ans = map(lambda x, t: x.astype(t), ans, outvar_dtypes)
        map(write, eqn.outvars, ans)
        clean_up_dead_vars(eqn, env, lu)
    ans = map(read, jaxpr.outvars)
    return ans




def amp(
    sentinal=None, *, compute_dtype=ml_dtypes.bfloat16, amp_policy=default_amp_policy, static_argnums=(), filter=True
):
    print("amp policy provided: ",amp_policy)
    if sentinal is not None:
        return amp(
                compute_dtype=compute_dtype,
                amp_policy=amp_policy,
                static_argnums=static_argnums,
                filter=filter)(sentinal)
    assert len(static_argnums) == 0 and filter, "currently only support filtering to find static arguments"

    def decorator(fn: Callable):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            flat_args, flat_args_treedef = jtu.tree_flatten((args, kwargs))

            array_idx = [i for i, x in enumerate(flat_args) if eqx.is_array(x)]
            static_idx = [i for i, x in enumerate(flat_args) if not eqx.is_array(x)]

            array_args = [flat_args[i] for i in array_idx]
            static_args = [flat_args[i] for i in static_idx]

            def flat_fn(*flat_args):
                args, kwargs = jtu.tree_unflatten(flat_args_treedef, flat_args)
                return fn(*args, **kwargs)

            jaxpr_generator = jax.make_jaxpr(flat_fn, static_argnums=static_idx, return_shape=True)

            closed_jaxpr, output_shape = jaxpr_generator(*flat_args)
            _, output_treedef = jtu.tree_flatten(output_shape)
            flat_out = amp_eval_jaxpr(compute_dtype, amp_policy, closed_jaxpr, *array_args)
            return jtu.tree_unflatten(output_treedef, flat_out)

        return wrapped_fn

    return decorator

