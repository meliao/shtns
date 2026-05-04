import jax
import ctypes

import numpy as np

import shtns_jax
import shtns
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # support float64


sh = shtns_jax.sht(127, 8)
sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)
print(
    "lmax=%d, mmax=%d, mres=%d, nlm=%d; nlat=%d, nphi=%d"
    % (sh.lmax, sh.mmax, sh.mres, sh.nlm, sh.nlat, sh.nphi)
)


Nbatch = 8
qlm = np.zeros((Nbatch, sh.nlm), dtype=np.complex128)
for i in range(Nbatch):
    qlm[i, i] = i
print(qlm.shape, qlm.dtype)

qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
print(qlm_jax.shape, qlm_jax.dtype, qlm_jax.devices())

q_jax = sh.synth_jax(qlm_jax)  # call from jax, automatic batching
print("After first call, here are shape, dtype, devices:")
print(q_jax.shape, q_jax.dtype, q_jax.devices())

q_jax = jax.vmap(sh.synth_jax)(qlm_jax)  # also automatic vmap
print("After vmap, shape and dtype:")
print(q_jax.shape, q_jax.dtype)

synth_jitted = jax.jit(sh.synth_jax)
q_jax = synth_jitted(qlm_jax)  # also jit
print("After JIT, shape and dtype:")
print(q_jax.shape, q_jax.dtype)


q_ref = np.zeros((Nbatch, sh.nlat, sh.nphi))
for i in range(Nbatch):
    q_ref[i] = sh.synth(qlm[i])

print(q_ref.shape, q_jax.shape, "error:", jnp.amax(abs(q_jax - q_ref)))
