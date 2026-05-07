import jax

import numpy as np

import shtns
import shtns_jax

import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # support float64


sh = shtns_jax.sht(10, 10, 1)
sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)


Nbatch = 4
rng = np.random.default_rng(0)
qlm = rng.standard_normal((Nbatch, sh.nlm)) + 1j * rng.standard_normal((Nbatch, sh.nlm))

qlm_cp = qlm.copy()
q_ref_lst = []
for i in range(Nbatch):
    q_ref_lst.append(sh.synth(qlm_cp[i]))
q_ref = np.stack(q_ref_lst)

qlm_jax = jnp.array(qlm, dtype=jnp.complex128)

q_jax = sh.synth_jax(qlm_jax)  # call from jax, automatic batching
error_first_call = jnp.max(jnp.abs(q_jax - q_ref))
print("Error after first call:", error_first_call)

q_jax = jax.vmap(sh.synth_jax)(qlm_jax)  # also automatic vmap
error_vmap = jnp.amax(abs(q_jax - q_ref))
print("Error after vmap:", error_vmap)

synth_jitted = jax.jit(sh.synth_jax)
q_jax = synth_jitted(qlm_jax)  # also jit
error_jit = jnp.amax(abs(q_jax - q_ref))
print("Error after JIT:", error_jit)
