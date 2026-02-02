import jax
import ctypes

import numpy as np

import shtns
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)  # support float64


shtns_jax_lib = ctypes.cdll.LoadLibrary("./libshtns_jax.so")
print("JAX loads shtns version:")
shtns_jax_lib.shtns_print_version()

jax.ffi.register_ffi_target(
    "shtns_synth", jax.ffi.pycapsule(shtns_jax_lib.synth_cpu), platform="cpu")

try:
	jax.ffi.register_ffi_target(
		"shtns_synth_gpu", jax.ffi.pycapsule(shtns_jax_lib.synth_gpu), platform="cuda")
	jax.ffi.register_ffi_target(
		"shtns_synth_gpu_float", jax.ffi.pycapsule(shtns_jax_lib.synth_gpu_float), platform="cuda")
except:
	print("no gpu implem")


def shtns_synth_jax(self, x):
	if x.dtype != jnp.float64:
		print(x.dtype)
		raise ValueError("Only the float64 dtype is implemented by shtns")
	out_shape = (self.nlat, self.nphi) if len(x.shape) == 1 else (*x.shape[:-1], self.nlat, self.nphi)

	#call = jax.ffi.ffi_call("shtns_synth",    # target name, same as in jax.ffi.register_ffi_target() above
	#	jax.ShapeDtypeStruct(out_shape, jnp.float64), # shape and dtype of the output
	#	vmap_method="broadcast_all",  #The `vmap_method` parameter controls this function's behavior under `vmap`
	#)

	def get_impl(target_name):
		return lambda x: jax.ffi.ffi_call(target_name,    # target name, same as in jax.ffi.register_ffi_target() above
		  jax.ShapeDtypeStruct(out_shape, jnp.float64), # shape and dtype of the output
		  vmap_method="broadcast_all",
		)(x, cfg=int(self.this))

	#return call(x, cfg=int(self.this))   # int(self.this) : pass the pointer to underlying C object exposed by swig
	return jax.lax.platform_dependent(x, cpu=get_impl("shtns_synth"), cuda=get_impl("shtns_synth_gpu"))

## add new jax method to existing class:
shtns.sht.synth_jax = shtns_synth_jax

sh = shtns.sht(127,8)
sh.set_grid(flags=shtns.SHT_ALLOW_GPU+shtns.SHT_PHI_CONTIGUOUS)
print("lmax=%d, mmax=%d, mres=%d, nlm=%d; nlat=%d, nphi=%d" % (sh.lmax, sh.mmax, sh.mres, sh.nlm,  sh.nlat, sh.nphi))


Nbatch = 8
qlm = np.zeros((Nbatch, sh.nlm), dtype=np.complex128)
for i in range(Nbatch):
	qlm[i,i] = i
print(qlm.shape, qlm.dtype)

qlm_jax = jnp.array(qlm.view(np.float64), jnp.float64)
print(qlm_jax.shape, qlm_jax.dtype)

q_jax = sh.synth_jax(qlm_jax)  # call from jax, automatic batching
print(q_jax.shape)
q_jax = jax.vmap(sh.synth_jax)(qlm_jax)   # also automatic vmap
print(q_jax.shape)
q_jax = jax.jit(sh.synth_jax)(qlm_jax)   # also jit
print(q_jax.shape)


q_ref = np.zeros((Nbatch, sh.nlat, sh.nphi))
for i in range(Nbatch):
	q_ref[i] = sh.synth(qlm[i])

print(q_ref.shape, q_jax.shape, "error:", jnp.amax(abs(q_jax-q_ref)))

