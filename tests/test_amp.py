import pytest
import equinox as eqx
from equinox import nn
from jaxtyping import PRNGKeyArray
from jaxamp import (
    amp,
    DynamicScalerState,
    dynamic_scale_grad,
    dynamic_scale_value_and_grad,
    amp_stop,
    default_amp_policy,
    use_original_precision,
)
import jax
from jax import numpy as jnp
from jax.random import PRNGKey


class HalfLinear(eqx.Module):
    l1: nn.Linear

    def __init__(self, in_features, out_features, *, key):
        self.l1 = nn.Linear(in_features, out_features, use_bias=True, key=key)

    def __call__(self, x, *, key=None):
        l1 = eqx.tree_at(
            lambda tree: (tree.weight, tree.bias),
            self.l1,
            replace_fn=lambda x: x.astype(jnp.float16),
        )
        x = x.astype(jnp.float16)
        y = l1(x)
        return y.astype(jnp.float32)


class StopBiasLinear(eqx.Module):
    l1: nn.Linear

    def __init__(self, in_features, out_features, *, key):
        self.l1 = nn.Linear(in_features, out_features, use_bias=True, key=key)

    def __call__(self, x, *, key=None):
        y = self.l1.weight @ x
        with amp_stop():
            y = y + self.l1.bias
        return y


class HalfWeightLinear(eqx.Module):
    l1: nn.Linear

    def __init__(self, in_features, out_features, *, key):
        self.l1 = nn.Linear(in_features, out_features, use_bias=True, key=key)
        # print("l1: ", self.l1)

    def __call__(self, x, *, key=None):
        l1 = eqx.tree_at(
            lambda tree: (tree.weight),
            self.l1,
            replace_fn=lambda x: x.astype(jnp.float16),
        )
        # print("self.l1 weight: ", self.l1.weight)
        # print("l1 weight: ", l1.weight)
        x = x.astype(jnp.float16)
        y = l1(x)
        return y.astype(jnp.float32)


def test_linear_amp():
    key = PRNGKey(0)
    half = HalfLinear(2, 10, key=key)  # HalfNet(key)
    full = nn.Linear(2, 10, use_bias=True, key=key)
    bias_stopped = StopBiasLinear(2, 10, key=key)
    half_weight = HalfWeightLinear(2, 10, key=key)
    # full = FullNet(key)
    amp_full = amp(full)
    amp_half = amp(half)
    amp_stopped = amp(amp_stop(full))
    amp_bias_stopped = amp(bias_stopped)
    amp_no_linear = amp(
        full, amp_policy=default_amp_policy | {"eqx.nn.Linear": use_original_precision}
    )

    x = jnp.array([5.0, 2.0])

    amp_stopped_out = amp_stopped(x)
    amp_bias_out = amp_bias_stopped(x)
    half_weight_out = half_weight(x)
    bias_out = bias_stopped(x)
    half_out = half(x)
    amp_full_out = amp_full(x)
    full_out = full(x)
    amp_half_out = amp_half(x)
    amp_no_linear_out = amp_no_linear(x)

    assert jnp.allclose(half_out, amp_full_out)
    assert jnp.allclose(half_out, amp_half_out)
    assert jnp.allclose(half_weight_out, amp_bias_out)
    assert jnp.allclose(bias_out, full_out)
    assert not jnp.allclose(full_out, half_out)
    assert jnp.allclose(full_out, amp_stopped_out)
    assert jnp.allclose(full_out, amp_no_linear_out)


def test_static():
    key = PRNGKey(0)
    bn, state = eqx.nn.make_with_state(eqx.nn.BatchNorm)(
        2, axis_name="batch"
    )  # , key=key)

    def get_batch_value(bn, state, data):
        ans, state = bn(data, state)
        return ans, state

    def get_value(bn, state, data):
        vmap_value = jax.vmap(
            get_batch_value,
            axis_name="batch",
            in_axes=(None, None, 0),
            out_axes=(0, None),
        )
        return vmap_value(bn, state, data)

    amp_get_value = amp(get_value)

    data = jnp.ones((3, 2, 4))
    value, _ = get_value(bn, state, data)
    amp_value, _ = amp_get_value(bn, state, data)

    # with all ones there will be no floating point error
    assert jnp.allclose(value, amp_value)


@pytest.mark.parametrize("tx", ["grad", "value_and_grad"])
@pytest.mark.parametrize("f", [True, False])
@pytest.mark.parametrize("aux", [True, False])
def test_dynamic_scaler(tx, f, aux):
    if tx == "grad":
        transform = dynamic_scale_grad
    elif tx == "value_and_grad":
        transform = dynamic_scale_value_and_grad

    def func(x):
        if aux:
            return x**3, 1.0
        return x**3

    x = jnp.array(2.0, dtype=jnp.float16)

    scaler_state = DynamicScalerState(scaler=2**15, patience=10)
    grad_fn = jax.jit(transform(func, filter=f, has_aux=aux, redo_on_nan=100))
    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    if tx == "value_and_grad" and aux:
        (v, a), g = g
    if tx == "value_and_grad" and not aux:
        v, g = g
    if tx == "grad" and aux:
        g, a = g
    if aux:
        assert jnp.allclose(a, 1.0)
    if tx == "value_and_grad":
        assert jnp.allclose(v, 8.0)
    assert jnp.allclose(g, 12.0)
    assert jnp.allclose(scaler_state.scaler, 2**12)
    assert jnp.allclose(scaler_state.count, 1.0)

    for i in range(9):
        scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
        assert jnp.allclose(scaler_state.scaler, 2**12)
        assert jnp.allclose(scaler_state.count, i + 2.0)
        if tx == "value_and_grad" and aux:
            (v, a), g = g
        if tx == "value_and_grad" and not aux:
            v, g = g
        if tx == "grad" and aux:
            g, a = g

        if aux:
            assert jnp.allclose(a, 1.0)
        if tx == "value_and_grad":
            assert jnp.allclose(v, 8.0)

        assert jnp.allclose(g, 12.0)

    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**13)
    assert jnp.allclose(scaler_state.count, 0.0)

    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**12)
    assert jnp.allclose(scaler_state.count, 1.0)

    scaler_state = DynamicScalerState(scaler=2**15)
    grad_fn = jax.jit(transform(func, filter=f, has_aux=aux, redo_on_nan=0))
    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**14)
    assert jnp.allclose(scaler_state.count, 0.0)

    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**13)
    assert jnp.allclose(scaler_state.count, 0.0)

    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**12)
    assert jnp.allclose(scaler_state.count, 0.0)

    scaler_state, g = grad_fn(x, dynamic_scaler_state=scaler_state)
    assert jnp.allclose(scaler_state.scaler, 2**12)
    assert jnp.allclose(scaler_state.count, 1.0)

    if tx == "value_and_grad" and aux:
        (v, a), g = g
    if tx == "value_and_grad" and not aux:
        v, g = g
    if tx == "grad" and aux:
        g, a = g

    if aux:
        assert jnp.allclose(a, 1.0)
    if tx == "value_and_grad":
        assert jnp.allclose(v, 8.0)

    assert jnp.allclose(g, 12.0)
