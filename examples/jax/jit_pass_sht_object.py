"""Example of passing a shtns.sht object into a function compiled with jax.jit."""

import jax
import jax.numpy as jnp
import numpy as np
import shtns

jax.config.update("jax_enable_x64", True)


def main():
    sh = shtns.sht(8, 8, 1)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)

    rng = np.random.default_rng(0)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)

    def synth_with_cfg(cfg, x):
        return cfg.synth_jax(x)

    synth_jit = jax.jit(synth_with_cfg, static_argnums=0)

    cache_size_before = synth_jit._cache_size()

    out = synth_jit(sh, qlm_jax)
    out.block_until_ready()

    cache_size_after_first = synth_jit._cache_size()

    # Print the cache size
    print("Cache size before:", cache_size_before)
    print("Cache size after first:", cache_size_after_first)

    out2 = synth_jit(sh, qlm_jax)
    out2.block_until_ready()

    cache_size_after_second = synth_jit._cache_size()
    print("Cache size after second:", cache_size_after_second)

    assert cache_size_after_first == cache_size_before + 1, (
        "Unexpected cache size after first compile."
    )
    assert cache_size_after_second == cache_size_after_first, (
        "Unexpected recompilation with same static args."
    )

    # New sht object with new Lmax, should trigger recompilation
    sh2 = shtns.sht(10, 10, 1)
    sh2.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)

    qlm_2 = rng.standard_normal(sh2.nlm) + 1j * rng.standard_normal(sh2.nlm)
    qlm_2_jax = jnp.array(qlm_2, dtype=jnp.complex128)

    out3 = synth_jit(sh2, qlm_2_jax)
    out3.block_until_ready()

    cache_size_after_third = synth_jit._cache_size()
    print("Cache size after third:", cache_size_after_third)
    assert cache_size_after_third == cache_size_after_second + 1, (
        "Expected cache size to increase."
    )


if __name__ == "__main__":
    main()
