import jax
import jax.numpy as jnp
from functools import wraps

from jax import lax
from jax._src.util import (
    safe_map,
)
from jax._src.core import (
    Jaxpr,
    Atom,
    Literal,
    Var,
    last_used,
    clean_up_dead_vars,
)
from jax._src import source_info_util
from typing import (
    Any,
    Callable,
    Sequence,
    Type,
    Dict,
    Optional,
)
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
    compute_dtype: Type,
    original_dtypes: Sequence[Type],
    *invars: Sequence[Any],
    **bind_params: Dict[str, Any],
) -> (Sequence[Any], Dict[str, Any]):
    return [
        cast_if_float(dtype, inv) for inv, dtype in zip(invars, original_dtypes)
    ], bind_params


def use_widest_precision(
    compute_dtype: Type,
    original_dtypes: Sequence[Type],
    *invars: Sequence[Any],
    **bind_params: Dict[str, Any],
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
    compute_dtype: Type,
    original_dtypes: Sequence[Type],
    *invars: Sequence[Any],
    **bind_params: Dict[str, Any],
) -> (Sequence[Any], Dict[str, Any]):
    invars = cast_tree(compute_dtype, invars)
    bind_params = cast_tree(compute_dtype, bind_params)
    bind_params = dict(bind_params)
    if "preferred_element_type" in bind_params:
        bind_params["preferred_element_type"] = compute_dtype
    return invars, bind_params


low_precision_primitives = [
    lax.add_p,
    lax.sub_p,
    lax.dot_general_p,
    lax.conv_general_dilated_p,
]

high_precision_primitives = [
    lax.exp_p,
    lax.log_p,
    lax.mul_p,
    lax.div_p,
]

# this dict specifies how different ops should be treated
# for AMP. Right now it is very simple: we move to low
# precision for additions, subtractions and
# tensor contractions  (e.g. matrix multiplies, convs etc)
# and high precision for exponentiation, logarithm, multiplication
# and division.
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
    compute_dtype: Type, amp_policy: Dict[Any, Any], closed_jaxpr: Jaxpr, *args
) -> Any:
    jaxpr = closed_jaxpr.jaxpr

    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        env[v] = val

    def map_type(x, t):
        if eqx.is_array(x):
            return x.astype(t)
        return x

    def restore_type(ans, outvars):
        out_types = map(lambda x: x.aval.dtype, outvars)
        return map(map_type, ans, out_types)

    def scope_in_policy(s):
        if isinstance(s, str):
            return any(*[s in p for p in amp_policy if isinstance(p, str)])
        else:
            return s in amp_policy

    env: dict[Var, Any] = {}
    map(write, jaxpr.constvars, closed_jaxpr.consts)
    map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        raw_name_stack = [s.name for s in name_stack.stack]
        traceback = eqn.source_info.traceback
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = map(read, eqn.invars)
            raw_name_stack.append(eqn.primitive)
            raw_name_stack.append("amp_default")
            for scope in raw_name_stack:
                if scope in amp_policy:
                    invar_dtypes = map(lambda x: x.aval.dtype, eqn.invars)
                    invars, bind_params = amp_policy[scope](
                        compute_dtype, invar_dtypes, *invars, **bind_params
                    )
                    break
            ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
            if not eqn.primitive.multiple_results:
                ans = [ans]
        map(write, eqn.outvars, ans)
        clean_up_dead_vars(eqn, env, lu)
    ans = map(read, jaxpr.outvars)
    ans = restore_type(ans, jaxpr.outvars)
    return ans


def amp(
    sentinal: Optional[Callable] = None,
    *,
    compute_dtype: Type = jnp.float16,
    amp_policy: Dict = default_amp_policy,
) -> Callable:
    if sentinal is not None:
        return amp(compute_dtype=compute_dtype, amp_policy=amp_policy)(sentinal)

    def decorator(fn: Callable) -> Any:
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            def _is_struct(x):
                return eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)

            flat_dynamic_args, _ = jtu.tree_flatten(
                eqx.filter((args, kwargs), _is_struct)
            )

            jaxpr_generator = eqx.filter_make_jaxpr(fn)

            closed_jaxpr, output_shape, static_out = jaxpr_generator(*args, **kwargs)

            _, output_treedef = jtu.tree_flatten(output_shape)
            flat_out = amp_eval_jaxpr(
                compute_dtype, amp_policy, closed_jaxpr, *flat_dynamic_args
            )
            dynamic_out = jtu.tree_unflatten(output_treedef, flat_out)
            ans = eqx.combine(dynamic_out, static_out)
            return ans

        return wrapped_fn

    return decorator
