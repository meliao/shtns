# JAX implementation notes


## Installation instructions

Build the JAX library with 
```
pip install ".[jax]" --no-build-isolation
```

This builds two modules, the `shtns` module we all know and love, and a second module `shtns_jax`. This second module extends the definition of the `shtns.sht` object to include JAX-compatible transforms:

| Transform | JAX CPU | JAX CUDA | JAX autodiff (CPU) | JAX autodiff (CUDA) | Notes |
| --- | --- | --- | --- | --- | --- |
| `synth_jax` | Y | Y | Y | Y | |
| `analys_jax` | Y | Y | Y | Y | |
| `synth_cplx_jax` | N | N | N | N | |
| `analys_cplx_jax` | N | N | N | N | |
| `synth_vec_jax` | Y | N | N | N | Vectorial SH -> spat transform. |
| `analys_vec_jax` | Y | N | N | N | Vectorial spat -> SH transform. |


## Testing the JAX code

Test it on the command line via
```
python -m pytest tests/
```
This requires having the pytest package installed.

## Usage examples

See the `examples/jax/` folder for a few usage examples.