import pytest
import equinox as eqx
from equinox import nn
from jaxtyping import Array, PRNGKeyArray
from eqxamp import MixedTypes, amp, DynamicScalarState, dynamic_scale_grad, dynamic_scale_value_and_grad
import jax
from jax import numpy as jnp
from jax.random import PRNGKey, split
from jax import tree_util as jtu
import ml_dtypes
from typing import Callable


class MP_Linear(nn.Linear):
    types: MixedTypes = eqx.field(static=True)
    square: Callable

    def __init__(self, mp_types: MixedTypes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.types = mp_types
        self.square = jtu.Partial(lambda x: x**2)

        self.weight = self.weight.astype(self.types.storage_type)
        if self.use_bias:
            self.bias = self.bias.astype(self.types.storage_type)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None) -> Array:
        if self.use_bias:
            where = lambda tree: (tree.weight, tree.bias)
        else:
            where = lambda tree: tree.weight

        compute_module = eqx.tree_at(
            where=where, pytree=self, replace_fn=lambda node: node.astype(self.types.compute_type)
        )

        x = x.astype(self.types.compute_type)
        result = nn.Linear.__call__(compute_module, x)
        print("result dtype: ", result.dtype)
        result = self.square(result)
        return result.astype(self.types.output_type)


class HalfLinearLN(eqx.Module):
    l1: MP_Linear
    ln: nn.LayerNorm

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray):
        type = MixedTypes(jnp.float32, ml_dtypes.bfloat16, jnp.float32)
        self.l1 = MP_Linear(type, in_features, out_features, key=key)
        self.ln = nn.LayerNorm(out_features)

    def __call__(self, x, *, key=None):
        x = self.l1(x)
        x = self.ln(x)
        return x


class HalfNet(eqx.Module):
    l1: MP_Linear
    h: HalfLinearLN
    l2: MP_Linear

    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        type = MixedTypes(jnp.float32, ml_dtypes.bfloat16, jnp.float32)
        self.l1 = MP_Linear(type, 1, 10, key=keys[0])
        self.h = HalfLinearLN(10, 10, key=keys[1])
        self.l2 = MP_Linear(type, 10, 1, key=keys[2])

    @jax.jit
    def __call__(self, x, *, key=None):
        x = self.l1(x, key=key)
        x = self.h(x, key=key)
        x = self.l2(x, key=key)
        return x


class FullLinearLN(eqx.Module):
    l1: nn.Linear
    ln: nn.LayerNorm
    square: Callable

    def __init__(self, in_features: int, out_features: int, *, key: PRNGKeyArray):
        self.l1 = nn.Linear(in_features, out_features, key=key)
        self.ln = nn.LayerNorm(out_features)
        self.square = jtu.Partial(lambda x: x**2)

    def __call__(self, x):
        x = self.l1(x)
        x = self.square(x)
        x = self.ln(x)
        return x


class FullNet(eqx.Module):
    l1: nn.Linear
    square1: Callable
    h: FullLinearLN
    l2: nn.Linear
    square2: Callable

    def __init__(self, key: PRNGKeyArray):
        keys = split(key, 3)
        self.l1 = nn.Linear(1, 10, key=keys[0])
        self.h = FullLinearLN(10, 10, key=keys[1])
        self.l2 = nn.Linear(10, 1, key=keys[2])
        self.square1 = jtu.Partial(lambda x: x**2)
        self.square2 = jtu.Partial(lambda x: x**2)

    @jax.jit
    def __call__(self, x, key=None):
        x = self.l1(x)
        x = self.square1(x)
        x = self.h(x)
        x = self.l2(x)
        x = self.square2(x)
        return x


def test_amp():
    key = PRNGKey(0)
    half = HalfNet(key)
    full = FullNet(key)
    amp_full = amp(full)
    amp_half = amp(half)

    x = jnp.array([5.0])

    half_out = half(x)
    amp_full_out = amp_full(x)
    full_out = full(x)
    amp_half_out = amp_half(x)

    print(half_out)
    print(amp_full_out)

    assert jnp.allclose(half_out, amp_full_out)
    assert jnp.allclose(half_out, amp_half_out)
    assert not jnp.allclose(full_out, half_out)

@pytest.mark.parametrize("tx", ["grad", "value_and_grad"])
@pytest.mark.parametrize("f", [True, False])
@pytest.mark.parametrize("aux", [True, False])
def test_dynamic_scalar(tx, f, aux):

    if tx == "grad":
        transform = dynamic_scale_grad
    elif tx == "value_and_grad":
        transform = dynamic_scale_value_and_grad
    def func(x):
        if aux:
            return x**3, 1.0
        return x**3

    x = jnp.array(2.0, dtype=jnp.float16)

    scalar_state = DynamicScalarState(scalar=2**15, patience=10)
    grad_fn = jax.jit(transform(func, filter=f, has_aux=aux, redo_on_nan=100))
    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
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
    assert jnp.allclose(scalar_state.scalar, 2**12)
    assert jnp.allclose(scalar_state.count, 1.0)

    for i in range(9):
        scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
        assert jnp.allclose(scalar_state.scalar, 2**12)
        assert jnp.allclose(scalar_state.count, i + 2.0)
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

    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**13)
    assert jnp.allclose(scalar_state.count, 0.0)

    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**12)
    assert jnp.allclose(scalar_state.count, 1.0)

    scalar_state = DynamicScalarState(scalar=2**15)
    grad_fn = jax.jit(transform(func, filter=f, has_aux=aux, redo_on_nan=0))
    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**14)
    assert jnp.allclose(scalar_state.count, 0.0)

    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**13)
    assert jnp.allclose(scalar_state.count, 0.0)

    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**12)
    assert jnp.allclose(scalar_state.count, 0.0)

    scalar_state, g = grad_fn(x, dynamic_scalar_state=scalar_state)
    assert jnp.allclose(scalar_state.scalar, 2**12)
    assert jnp.allclose(scalar_state.count, 1.0)

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
