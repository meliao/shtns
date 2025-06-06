import jax
import ctypes

import numpy as np

import shtns
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)  # support float64


shtns_jax_lib = ctypes.cdll.LoadLibrary("./libshtns_jax.so")
jax.ffi.register_ffi_target(
    "shtns_synth", jax.ffi.pycapsule(shtns_jax_lib.synth_cpu), platform="cpu")
print("JAX loads shtns version:")
shtns_jax_lib.shtns_print_version()

def shtns_synth(x, sh=0):
	if x.dtype != jnp.float64:
		print(x.dtype)
		raise ValueError("Only the float64 dtype is implemented by shtns")
	out_shape = (sh.nlat, sh.nphi) if len(x.shape) == 1 else (*x.shape[:-1], sh.nlat, sh.nphi)
	call = jax.ffi.ffi_call("shtns_synth",    # target name, same as in jax.ffi.register_ffi_target() above
		jax.ShapeDtypeStruct(out_shape, jnp.float64), # shape and dtype of the output
		vmap_method="broadcast_all",  #The `vmap_method` parameter controls this function's behavior under `vmap`
	)
	return call(x, cfg=int(sh.this))   # int(sh.this) : pass the pointer to underlying C object


sh = shtns.sht(127,8)
sh.set_grid()


Nbatch = 8
qlm = np.zeros((Nbatch, sh.nlm), dtype=np.complex128)
for i in range(Nbatch):
	qlm[i,i] = i
print(qlm.shape, qlm.dtype)

qlm_jax = jnp.array(qlm.view(np.float64), jnp.float64)
print(qlm_jax.shape, qlm_jax.dtype)

q_jax = shtns_synth(qlm_jax, sh=sh)  # call from jax, automatic batching
#q_jax = jax.vmap(shtns_synth)(qlm_jax, sh=sh)

q_ref = np.zeros((Nbatch, sh.nlat, sh.nphi))
for i in range(Nbatch):
	q_ref[i] = sh.synth(qlm[i])

print(q_ref.shape, q_jax.shape, jnp.amax(abs(q_jax-q_ref)))

