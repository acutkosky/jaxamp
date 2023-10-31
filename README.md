# jaxamp: automatic mixed precision in JAX

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install jaxamp
```

## Usage

TL;DR: Like [pytorch amp](https://pytorch.org/docs/stable/amp.html), but for JAX.

Replace `loss_fn(model, minibatch)` with `jaxamp.amp(loss_fn)(model,minibatch)` to run with with [mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). Use `scaler_state = jaxamp.DynamicScalerState()` and `jaxamp.dynamic_scale_grad` or `jaxamp.dynamic_scale_value_and_grad` to apply a dynamic loss scaler:
```python
def loss(model, minibatch):
  ...

scaler_state= jaxamp.DynamicLossScaler()
amp_loss = jaxamp.amp(loss)
grad_fn = jaxamp.dynamic_scale_grad(amp_loss)
scaler_state, grad = grad_fn(model, minibatch, dynamic_scaler_state=scaler_state)
```



### More details
Your usual training loop might look like this:
```python

def loss_fn(model_state, minibatch):
  ...
  return loss, accuracy

def train_step(model_state, opt_state, minibatch, optimizer):

  value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

  (loss, accuracy), grads = value_and_grad_fn(model_state, minibatch)

  updates, opt_state = optimizer.update(grads, opt_state, model_state)
  model_state = optax.apply_updates(model_state, updates)
  return model_state, opt_state, loss, accuracy

def train_loop(model_state, opt_state, optimizer, dataloader):
  train_step_jit = jax.jit(train_step, static_argnums=3)

  for minibatch in dataloader:
    model_state, opt_state, loss, accuracy = train_step_jit(model_state, opt_state, minibatch, optimizer)
    log_metrics(loss, accuracy)
  return model_state, opt_state
```


Now, you can replace this with:
```python

def train_step(
    model_state,
    opt_state,
    minibatch,
    dynamic_scaler_state,
    optimizer):
  amp_loss_fn = jaxamp.amp(loss_fn)
  
  value_and_grad_fn = jaxamp.dynamic_scale_value_and_grad(amp_loss_fn, has_aux=True)

  dynamic_scaler_state, ((loss, accuracy), grads) = value_and_grad_fn(
    model_state,
    minibatch,
    dynamic_scaler_state=dynamic_scaler_state)

  updates, opt_state = optimizer.update(grads, opt_state, model_state)
  model_state = optax.apply_updates(model_state, updates)
  return model_state, opt_state, dynamic_scaler_state, loss, accuracy

def train_loop(model_state, opt_state, optimizer, dataloader):
  train_step_jit = jax.jit(train_step, static_argnums=3)
  dynamic_scaler_state = amp.DynamicScalerState()
  for minibatch in dataloader:
    model_state, opt_state, dynamic_scaler_state, loss, accuracy = train_step_jit(
      model_state,
      opt_state,
      minibatch,
      optimizer)
    log_metrics(loss, accuracy)
  return model_state, opt_state
```

It should now be faster!

### More details on `amp`

The `amp` function transforms an arbitrary function into one in which some operations are performed in low precision. This precision can be controlled via the `compute_dtype` keyword-only argument:
`amp_loss_fn = amp(loss_fn, compute_dtype=jnp.float16)`. You can also control which operations are performed in low precision (and how) via the `amp_policy` keyword-only argument. This argument should
take a dictionary whose keys must be either strings or jax primitives (e.g. `jax.lax.add_p`). The values are functions that will be called to cast arrays into relevant dtypes. These functions should have signature:
```python
def precision_fn(
    compute_dtype: Type,
    original_dtypes: Sequence[Type],
    *invars: Sequence[Array],
    *bind_params: Dict[str, Any]) -> Sequence[Array], Dict[str, Any]:
  '''
  Args:
    compute_dtype: this is the compute_dtype provided to `amp`.
    original_dtypes: these are the dtypes that original user code expected the arguments
        to the op we are about to transform were going  to be.
    invars: the input arrays to this operation (note that these dtypes may not match
        original_dtypes because of previous casting we might have performed).
    bind_params: the "meta" parameters to the op (things like axis specifications).
  returns
    new_invars, new_bind_params: the transformed values for invars and bind_params.
  '''
```
For example, the function used to cast to `compute_dtype` is:
```python
def use_compute_precision(
    compute_dtype: Type,
    original_dtypes: Sequence[Type],
    *invars: Sequence[Any],
    **bind_params: Dict[str, Any]
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
```

`amp` will walk through all the ops in your function and look up each op in your `amp_policy` dict. If the op is present, it will apply the specified function
Otherwise it  will cast the inputs to their original values and apply the op unchanged. You can also provide string keys in `amp_policy`. In this case, if the current operation
is executed inside a scope declared with `jax.named_scope`, we will apply the  specified transformation function. If two or more active scopes match policies in `amp_policy` the *outermost* scope is used. There are two special scopes `"amp_step"` and `"amp_default"`.
By default these both stop any automatic mixed precision from happening inside them.


### More details on dynamic loss scalers

We supply a loss scaling operation via `DynamicScalerState` and corresponding functions `dynamic_scale_grad` and `dynamic_scale_value_and_grad`.

`DynamicScalerState` has the following structure:
```python
class DynamicScalerState(NamedTuple):
    patience: jax.Array = jnp.array(2000) # number of non-inf/NaN iterations to wait before increasing the scaler
    adjust_factor: jax.Array = jnp.array(2.0) # When increasing or decreasing the scaler, multiply or divide by this factor.
    scaler: jax.Array = jnp.array(2**15, dtype=jnp.float32) # current scaler value
    count: jax.Array = jnp.array(0) # number of non-inf/NaN iterations since the scaler was last increased.
```

The gradient functions then have behavior like:
```python
def dynamic_scale_value_and_grad(
    fun: Callable,
    *,
    has_aux: bool = False,
    redo_on_nan: bool = 0,
    filter=True,
    **kwargs
):
    '''
    apply dynamic scalar to the value_and_grad function.

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
```

### More usage tips

When using `optax`, you may want to wrap your optimizers in `optax.apply_if_finite` to automatically skip NaN gradients. Alternatively, you could use the `redo_on_nan` option.

## License

`jaxamp` is distributed under the terms of the [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html) license.
