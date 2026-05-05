import jax

import numpy as np

import shtns
import shtns_jax

import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # support float64


sh = shtns_jax.sht(10, 10, 1)
sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)
sh2 = shtns.sht(10 , 10, 1)
sh2.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)
print(
    "lmax=%d, mmax=%d, mres=%d, nlm=%d; nlat=%d, nphi=%d"
    % (sh.lmax, sh.mmax, sh.mres, sh.nlm, sh.nlat, sh.nphi)
)


Nbatch = 2
rng = np.random.default_rng(0)
qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
print(qlm.shape, qlm.dtype)

qlm_cp = qlm.copy()
q_ref = sh2.synth(qlm_cp)  # call from numpy, reference result
print("Nans in reference?", np.isnan(q_ref).any())

qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
print(qlm_jax.shape, qlm_jax.dtype, qlm_jax.devices())
print("Nans in input?", jnp.isnan(qlm_jax).any())

q_jax = sh.synth_jax(qlm_jax)  # call from jax, automatic batching
print("After first call, here are shape, dtype, devices:")
print(q_jax.shape, q_jax.dtype, q_jax.devices())
error_first_call = jnp.max(jnp.abs(q_jax - q_ref))
print("Error after first call:", error_first_call)

# q_jax = jax.vmap(sh.synth_jax)(qlm_jax)  # also automatic vmap
# print("After vmap, shape and dtype:")
# print(q_jax.shape, q_jax.dtype)
# error_vmap = jnp.amax(abs(q_jax - q_ref))
# print("Error after vmap:", error_vmap)

synth_jitted = jax.jit(sh.synth_jax)
q_jax = synth_jitted(qlm_jax)  # also jit
print("After JIT, shape and dtype:")
print(q_jax.shape, q_jax.dtype)
error_jit = jnp.amax(abs(q_jax - q_ref))
print("Error after JIT:", error_jit)
