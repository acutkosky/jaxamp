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


def default_precision(compute_dtype, *invars, **bind_params):
    return invars, bind_params


@curry
def cast_if_float(dtype, value):
    if eqx.is_inexact_array(value):
        return value.astype(dtype)
    return value


def low_precision(compute_dtype, *invars, **bind_params):
    # print("running low precision")
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
default_amp_policy = defaultdict(
    lambda: default_precision, # default value
    {
        op: low_precision for op in low_precision_primitives
    },
)



def amp_stop(f = None):
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
        # nonlocal env
        env[v] = val

    env: dict[Var, Any] = {}
    map(write, jaxpr.constvars, closed_jaxpr.consts)
    map(write, jaxpr.invars, args)
    # print(jaxpr.invars)
    # print("args: ", args)
    # print("env initial: ", env)
    lu = last_used(jaxpr)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = map(read, eqn.invars)
            # print(f"name stack: ",name_stack)
            names_in_stack = str(name_stack).split("/")#[elem.name for elem in name_stack]
            # print(f"eqn: {eqn}")
            if "amp_stop" not in names_in_stack:
                invars, bind_params = amp_policy[eqn.primitive](compute_dtype, *invars, **bind_params)
            # print(f"invars: {invars}")
            # print(f"subfuns: ",subfuns)
            # print(f"binparams: {bind_params}")
            # bind_params['preferred_element_type'] = compute_dtype
            outvar_dtypes = map(lambda x: x.aval.dtype, eqn.outvars)
            ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
            # print("outvar types: ",outvar_dtypes)
            ans = map(lambda x, t: x.astype(t), ans, outvar_dtypes)
        map(write, eqn.outvars, ans)
        clean_up_dead_vars(eqn, env, lu)
    ans = map(read, jaxpr.outvars)
    return ans



def amp(sentinal=None, *, compute_dtype=ml_dtypes.bfloat16, amp_policy=default_amp_policy, static_argnums=()):
    if sentinal is not None:
        return amp()(sentinal)

    def decorator(fn: Callable):

        jaxpr_generator = jax.make_jaxpr(fn, static_argnums=static_argnums, return_shape=True)

        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            flat_args, flat_args_treedef = jtu.tree_flatten((args, kwargs))
            closed_jaxpr, output_shape = jaxpr_generator(*args, **kwargs)
            _, output_treedef = jtu.tree_flatten(output_shape)
            flat_out = amp_eval_jaxpr(compute_dtype, amp_policy, closed_jaxpr, *flat_args)
            return jtu.tree_unflatten(output_treedef, flat_out)

        return wrapped_fn

    return decorator


# def blah(z, y, x):
#     return x, z, y


# d = {'y': 1, 'x': 2}
# print(list(d.keys()))
# jaxpr = jax.make_jaxpr(blah)({'p': 3}, **d)
# print(jaxpr)


# @jax.named_scope("sq")
# def sq(x):
#     return x**2


# @jax.named_scope("itsame")
# def func(x):
#     print("x:   !", x.shape)
#     return 1.0 / sq(x)


# from jax.experimental.maps import xmap

# xmfunc = xmap(amp(func), in_axes={0: 'batch'}, out_axes={0: 'batch'})

# p = xmfunc(2 * jnp.ones((3, 4)))

# print("xmfunc: ", p)


# ampfunc = jax.jit(amp(func))
# k = jax.random.PRNGKey(0)
# l = eqx.nn.Linear(5, 5, key=k)

# x = jnp.ones(5)

# closed_jaxpr = jax.make_jaxpr(l)(x)
# a = jax.jit(amp(l))(x)
# # a = amp_eval_jaxpr(closed_jaxpr, x)
# print("linear a: ", a)
# a = l(x)
# print("linear no amp a: ", a)

# x = jnp.array(1e-5)
# closed_jaxpr = jax.make_jaxpr(func)(x)
# # jnp.ones(5, dtype=jnp.float16))
# print(closed_jaxpr)
# print(closed_jaxpr.jaxpr)
# # a = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, x)#jnp.ones(5, dtype=jnp.float16))
# # a = amp_eval_jaxpr(closed_jaxpr, x)
# # jnp.ones(5, dtype=jnp.float16))
# # print(a)
# # print("lax mul_p: ", lax.mul_p)


# a = ampfunc(x)
# print("final a: ", a)
