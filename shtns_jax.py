### JAX interface for SHTns
### Requires jaxbind:  https://github.com/NIFTy-PPL/JAXbind
### Only CPU support for now.

### See also: https://docs.jax.dev/en/latest/external-callbacks.html#example-pure-callback-with-custom-jvp

### For adding GPU support to jaxbind, look at:
# https://github.com/jax-ml/jax/issues/1100#issuecomment-1876928289
# https://github.com/pearu/pydlpack

## TODO:
## 1) check that the adjoints are actually good
##    in particular, I use the inner-product corresponding to real-valued functions
##    there may be a factor of 2, or there may be complex conjugation subtelties...
##    We could also reformulate everything in terms of real arrays?



import jax
import jax.numpy as jnp
import shtns
import numpy as np

import jaxbind

jax.config.update("jax_enable_x64", True)

## IMPORTANT: shtns must be initialized BEFORE the use with JAX,
## and this configuration cannot be changed.
## Allowing multiple shtns configs could be added in the future if needed.

Lmax=47
shtns_cfg = shtns.sht(Lmax)     # single configuration used throughout
shtns_cfg.set_grid(nl_order=2)  # with a dealiased grid

## store the weights for adjoint stuff
shtns_cfg.wts = shtns_cfg.gauss_wts() * (2*np.pi)/shtns_cfg.nphi
shtns_cfg.wts = np.hstack((shtns_cfg.wts, np.flip(shtns_cfg.wts))).reshape((-1,1))   # works if nlat is even
shtns_cfg.wts_1 = 1.0 / shtns_cfg.wts
shtns_cfg._l2 = shtns_cfg.l * (shtns_cfg.l + 1.0)  # l(l+1)
shtns_cfg._l_2 = np.zeros_like(shtns_cfg._l2)
shtns_cfg._l_2[1:] = 1.0 / shtns_cfg._l2[1:]

def sht_synth(out, args, kwargs_dump):
    nargs = len(args)

    # deserialize keyword arguments
    #kwargs = jaxbind.load_kwargs(kwargs_dump)
    # extract keyword argument which can be given to the JAX primitive
    #workers = kwargs.pop("workers", None)

    # compute the SHT and write the result in the out tuple
    if nargs==1:
        out[0][()] = shtns_cfg.synth(args[0])
    else:
        q = shtns_cfg.synth(*args)
        for i in range(nargs):
            out[i][()] = q[i]


def sht_analys(out, args, kwargs_dump):
    nargs = len(args)
    if nargs==1:
        out[0][()] = shtns_cfg.analys(args[0])
    else:
        alm = shtns_cfg.analys(*args)
        for i in range(nargs):
            out[i][()] = alm[i]


def sht_synth_transposed(out, args, kwargs_dump):
    nargs = len(args)
    if nargs==1:
        out[0][()] = shtns_cfg.analys(args[0] * shtns_cfg.wts_1)
    elif nargs==2:
        slm,tlm = shtns_cfg.analys(args[0] * shtns_cfg.wts_1,  args[1] * shtns_cfg.wts_1)
        out[0][()] = slm * shtns_cfg._l2
        out[1][()] = tlm * shtns_cfg._l2
    elif nargs==3:
        qlm,slm,tlm = shtns_cfg.analys(args[0] * shtns_cfg.wts_1,  args[1] * shtns_cfg.wts_1,  args[2] * shtns_cfg.wts_1)
        out[0][()] = qlm
        out[1][()] = slm * shtns_cfg._l2
        out[2][()] = tlm * shtns_cfg._l2

def sht_analys_transposed(out, args, kwargs_dump):
    nargs = len(args)

    if nargs==1:
        out[0][()] = shtns_cfg.synth(args[0]) * shtns_cfg.wts
    elif nargs==2:
        u,v = shtns_cfg.synth(args[0] * shtns_cfg._l_2,  args[1] * shtns_cfg._l_2)
        out[0][()] = u * shtns_cfg.wts
        out[1][()] = v * shtns_cfg.wts
    elif nargs==3:
        q,u,v = shtns_cfg.synth(args[0], args[1] * shtns_cfg._l_2,  args[2] * shtns_cfg._l_2)
        out[0][()] = q * shtns_cfg.wts
        out[1][()] = u * shtns_cfg.wts
        out[2][()] = v * shtns_cfg.wts

def sht_synth_abstract_eval(*args, **kwargs):
    nargs = len(args)
    assert nargs <= 3
    for i in range(nargs):
        assert args[i].shape == (shtns_cfg.nlm,)
        assert args[i].dtype == jnp.complex128

    out_shape = (shtns_cfg.nlat, shtns_cfg.nphi)
    # return shape, dtype of output
    return ((out_shape, jnp.float64),) * nargs

def sht_analys_abstract_eval(*args, **kwargs):
    nargs = len(args)
    assert nargs <= 3
    for i in range(nargs):
        assert args[i].shape == (shtns_cfg.nlat, shtns_cfg.nphi)
        assert args[i].dtype == jnp.float64

    out_shape = (shtns_cfg.nlm,)
    # return shape, dtype of output
    return ((out_shape, jnp.complex128),) * nargs

# Now we register our function as a custom JAX primitive using JAXbind's
# interface for linear functions. JAXbind returns the resulting JAX primitive.

sht_synth_jax = jaxbind.get_linear_call(
    sht_synth,
    sht_synth_transposed,
    sht_synth_abstract_eval,
    sht_analys_abstract_eval,
    func_can_batch=False,  # indicate that our function DOES NOT support custom batching
)

sht_analys_jax = jaxbind.get_linear_call(
    sht_analys,
    sht_analys_transposed,
    sht_analys_abstract_eval,
    sht_synth_abstract_eval,
    func_can_batch=False,
)


# %%%%%%%%%%%%%%% Perform some tests %%%%%%%%%%%%%%%%

if __name__ == "__main__":

    from jax import random
    import numpy as np

    # generate some random input to showcase the use of the newly registered JAX primitive
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    qlm = jax.random.uniform(subkey, shape=(shtns_cfg.nlm, ), dtype=jnp.float64)
    qlm = qlm + 1j * jax.random.uniform(subkey, shape=(shtns_cfg.nlm, ), dtype=jnp.float64)


    # apply the new primitive
    res = sht_synth_jax(qlm)    # shtns call through JAX
    print(len(res), res[0].shape, qlm.shape)

    ref = shtns_cfg.synth(np.array(qlm))    # direct call to shtns
    print( np.allclose(res[0],ref) )   # compare results

    # jit compile the new primitive
    sht_synth_jax_jit = jax.jit(sht_synth_jax)
    res_jit = sht_synth_jax_jit(qlm)
    print(res_jit)
    print( np.allclose(res_jit[0],ref) )   # compare results

    # vmap sht_synth over the first axis of the input (first axis is the one that varies slowest in memory, so that the other axes are continuous)
    sht_synth_vmap = jax.vmap(sht_synth_jax, in_axes=0)

    qlm_batch = jnp.arange(1,7).reshape((-1,1)) * qlm    # make an array of several qlm fields, 

    res_batch = sht_synth_vmap(qlm_batch)
    print( res_batch[0].shape )

    for i in range(1,7):
        print( np.allclose(res_batch[0][i-1,:], ref*i) )   # compare results

    slm = qlm * 2
    res = sht_synth_jax(qlm,slm)   # transform of a 2D vector, res contains the two field (theta and phi components)
    print(res[0].shape, res[1].shape)


    # compute the jvp of sht_synth
    res_jvp = jax.jvp(sht_synth_jax, (qlm,), (qlm,))

