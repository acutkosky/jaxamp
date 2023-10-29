import jax
import jax.numpy as jnp
from functools import wraps

from jax import lax
from jax._src.util import (
    safe_map,
    curry,
)
from jax._src.core import Jaxpr, Atom, Literal, Var, last_used, clean_up_dead_vars
from jax._src import source_info_util
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Generic,
    NamedTuple,
    TypeVar,
    Union,
    Sequence,
    Type,
    Dict,
    Optional,
    cast,
    overload,
)
from types import MappingProxyType
from collections import defaultdict
import ml_dtypes
import equinox as eqx
from jaxtyping import PyTree
from jax import tree_util as jtu

map, unsafe_map = safe_map, map

precision_ordering = [
    jnp.float64,
    jnp.float32,
    ml_dtypes.bfloat16,
    jnp.float16,
]


def find_widest_dtype(dtype_list: Sequence[Type]) -> Type:
    for dtype in precision_ordering:
        if dtype in dtype_list:
            return dtype

    return dtype_list[0]


def use_original_precision(
    compute_dtype: Type, original_dtypes: Sequence[Type], *invars: Sequence[Any], **bind_params: Dict[str, Any]
) -> (Sequence[Any], Dict[str, Any]):
    return [cast_if_float(dtype, inv) for inv, dtype in zip(invars, original_dtypes)], bind_params


def use_widest_precision(
    compute_dtype: Type, original_dtypes: Sequence[Type], *invars: Sequence[Any], **bind_params: Dict[str, Any]
) -> (Sequence[Any], Dict[str, Any]):
    dtype = find_widest_dtype(original_dtypes)
    return use_compute_precision(dtype, original_dtypes, *invars, **bind_params)


def cast_if_float(dtype: Type, value: Any) -> Any:
    if eqx.is_inexact_array(value):
        if value.dtype != dtype:
            return value.astype(dtype)
    return value


def cast_tree(dtype: Type, tree: PyTree) -> PyTree:
    return jtu.tree_map(lambda v: cast_if_float(dtype, v), tree)


def use_precision(override_dtype: Type) -> Callable:
    def overridden_precision(compute_dtype, *args, **kwargs):
        return use_compute_precision(override_dtype, *args, **kwargs)

    return overridden_precision


def use_compute_precision(
    compute_dtype: Type, original_dtypes: Sequence[Type], *invars: Sequence[Any], **bind_params: Dict[str, Any]
) -> (Sequence[Any], Dict[str, Any]):
    for invar in invars:
        # if anything is an integer, let's not cast anything.
        if not eqx.is_inexact_array(invar):
            return invars, bind_params
    invars = cast_tree(compute_dtype, invars)
    bind_params = cast_tree(compute_dtype, bind_params)
    bind_params = dict(bind_params)
    if "preferred_element_type" in bind_params:
        bind_params["preferred_element_type"] = compute_dtype
    return invars, bind_params


low_precision_primitives = [
    lax.dot_general_p,
    lax.add_p,
    lax.sub_p,
    lax.conv_general_dilated_p,
    'eqx.nn.Linear',
    'eqx.nn.Conv2d',
    'eqx.nn.Conv',
]

high_precision_primitives = [
    lax.mul_p,
    lax.div_p,
    'eqx.nn.BatchNorm',
    'eqx.nn.LayerNorm',
    'eqx.nn.SpectralNorm',
]

# this dict specifies how different ops should be treated
# for AMP. Right now it is very simple: we move to low
# precision for additions, subtractions and
# tensor contractions  (e.g. matrix multiplies, convs etc).
# Otherwise we do not do anything.
default_amp_policy = (
    {op: use_compute_precision for op in low_precision_primitives}
    | {
        "amp_default": use_original_precision,
        "amp_stop": use_original_precision,
    }
    | {op: use_precision(jnp.float32) for op in high_precision_primitives}
)


def amp_stop(f: Callable = None):
    context = jax.named_scope("amp_stop")
    if f is None:
        return context
    else:
        return context(f)


def amp_eval_jaxpr(
    compute_dtype: Type, amp_policy: Dict[Any, Any], closed_jaxpr: Jaxpr, *args, propagate_source_info=True
) -> Any:
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
            scopes = str(name_stack).split("/")
            scopes.append(eqn.primitive)
            scopes.append("amp_default")
            for scope in scopes:
                if scope in amp_policy:
                    invar_dtypes = map(lambda x: x.aval.dtype, eqn.invars)
                    invars, bind_params = amp_policy[scope](compute_dtype, invar_dtypes, *invars, **bind_params)
                    break
            outvar_dtypes = map(lambda x: x.aval.dtype, eqn.outvars)
            ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
        map(write, eqn.outvars, ans)
        clean_up_dead_vars(eqn, env, lu)
    ans = map(read, jaxpr.outvars)
    ans_dtypes = map(lambda x: x.aval.dtype, jaxpr.outvars)
    def map_type(x, t):
        if eqx.is_array(x):
            return x.astype(t)
        return x
    ans = map(map_type, ans, ans_dtypes)
    return ans


def amp(
    sentinal: Optional[Callable] = None,
    *,
    compute_dtype: Type = jnp.float16,
    amp_policy: Dict = default_amp_policy,
) -> Callable:
    if sentinal is not None:
        return amp(compute_dtype=compute_dtype, amp_policy=amp_policy)(
            sentinal
        )

    def decorator(fn: Callable) -> Any:
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
            ans = jtu.tree_unflatten(output_treedef, flat_out)

            # For some reason, not including the following commented-out line
            # (which should be a no-op since the output is never accessed) causes jit to
            # recompile a few more times than it should really need to. Outside of a jit
            # context though, this line is very expensive. So, we'll leave it out
            # and just eat the small number of extra compiles.
            # It would be great to understand why those compiles are happening though...

            # nonjaxpr_ans = flat_fn(*flat_args)

            return ans

        return wrapped_fn

    return decorator
