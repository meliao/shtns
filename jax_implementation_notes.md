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
| `synth_vec_jax` | Y | Y | Y | N | Vectorial SH -> spat transform. |
| `analys_vec_jax` | Y | Y | Y | N | Vectorial spat -> SH transform. |
| `synth_vec_cplx_jax` | Y | N | Y | N | Complex vectorial SH -> spat transform. |
| `analys_vec_cplx_jax` | Y | N | Y | N | Complex vectorial spat -> SH transform. |
| `SHqst_to_point_cplx_jax` | N | N | N | N | Vector SH -> arbitrary point. We have a NumPy implementation in `shtns_jax.py`. |
| `SHqst_to_lat_jax` | Y | N | N | N | |


And we have done the same for thet `shtns.rotation` object. The definition is extended in the `shtns_jax.rotation` class to include these JAX-compatible transforms:

| Transform | JAX CPU | JAX CUDA | JAX autodiff (CPU) | JAX autodiff (CUDA) | Notes |
| --- | --- | --- | --- | --- | --- |
| `apply_real_jax` | Y | N | N | N | CPU Autodiff is implemented but not validated numerically. |
| `apply_cplx_jax` | Y | N | N | N | CPU Autodiff is implemented but not validated numerically. |

## Building with OpenMP support (macOS)

Apple's system `clang` (aliased as `gcc`/`g++` on PATH) does not support `-fopenmp`, so
`setup.py`'s OpenMP auto-detection silently falls back to a non-threaded build on macOS
unless a real OpenMP-capable compiler is pointed to explicitly. Install Homebrew GCC and
FFTW once (`brew install gcc fftw` — `fftw` provides `libfftw3_omp`), then rebuild from a
clean state with four environment variables set:

```bash
rm -f *.o Makefile sht_config.h config.status config.log libshtns_jax_cpu.so _shtns*.so
CC=/opt/homebrew/bin/gcc-15 CXX=/opt/homebrew/bin/g++-15 \
CFLAGS='-D__ARM_NEON_FP=0xE' LIBRARY_PATH=/opt/homebrew/lib \
  pip install ".[jax]" --no-build-isolation
```

Verified working (`import shtns_jax` prints `...,neon,ishioka,openmp` and
`otool -L` on both `_shtns*.so` and `libshtns_jax_cpu.so` shows `libfftw3_omp`/`libgomp`
linked). Each piece is required for a distinct reason:

- **`rm -f ...` (clean first)** — `make`'s dependency tracking only checks file
  timestamps, not which compiler/flags were used. Without removing the existing `.o`
  files, `Makefile`, and `.so` outputs, `make` may decide they're already up to date and
  skip recompiling them even though the compiler changed, silently keeping a stale build.
- **`CC`/`CXX` → Homebrew GCC** — needed so `setup.py`'s `check_openmp_support()` probe,
  `./configure`'s `AC_PROG_CC`, and the JAX FFI `.so` build all pick a compiler that
  actually accepts `-fopenmp`. Check `ls /opt/homebrew/bin/gcc-*` for the installed
  version — the `-15` suffix tracks Homebrew's GCC major version and needs bumping after
  a `brew upgrade gcc`.
- **`CFLAGS='-D__ARM_NEON_FP=0xE'`** — switching the main compiler to GCC has a side
  effect: `shtns_simd.h` enables NEON vector code via `#if _GCC_VEC_ && (__ARM_NEON_FP >=
  8)`, but on this platform Homebrew's GCC never predefines `__ARM_NEON_FP` (only
  `__ARM_NEON`/`__ARM_FP`), unlike Apple's clang. Without this define, any file compiled
  by the main `cc` (notably `sht_init.o` and — critically — `sht_omp.o`, the actual
  OpenMP-parallel kernel) silently falls back to scalar (non-vectorized) code, which can
  undercut the benefit of enabling threading in the first place. The separate kernel
  compiler (`shtcc`, used for `sht_fly.o`/`sht_kernels_*.o`) stays on Apple clang and is
  unaffected either way.
- **`LIBRARY_PATH=/opt/homebrew/lib`** — `check_openmp_support()`'s test link step does
  `-lfftw3_omp` with no `-L` path, so without this the linker fails with
  `ld: library 'fftw3_omp' not found` (Homebrew's `libfftw3_omp.dylib` lives under
  `/opt/homebrew/lib`) and OpenMP silently never gets enabled, independent of the NEON
  issue above.

`SHTNS_OPENMP=0` can still be set to force a non-threaded build regardless of the above.

Also watch for **stale repo-root artifacts shadowing the installed package**: if
`_shtns*.so`/`shtns.py`/`libshtns_jax_cpu.so` exist in the repo root (e.g. from an earlier
manual `setup.py build_ext --inplace`), running `python` from inside the repo directory
will import those instead of the freshly-installed ones in `site-packages`, because the
current directory comes first on `sys.path`. Delete the repo-root copies (they're
gitignored build artifacts, safe to remove) if the version banner doesn't reflect a
rebuild.

## Testing the JAX code

Test it on the command line via
```
python -m pytest tests/
```
This requires having the pytest package installed.

## Usage examples

See the `examples/jax/` folder for a few usage examples.

## Known issues

1. `sht.m` and `sht.l` arrays are saved as 32-bit integers; when JAX operates on them, the result is often cast to `jnp.float32` which can be an issue.