import numpy as np
import pytest

import shtns


def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    return sh


@pytest.mark.parametrize("contig", [True, False])
@pytest.mark.parametrize("n_args", [1, 2])
def test_synth_does_not_modify_inputs(contig, n_args):
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(123)

    inputs = []
    for _ in range(n_args):
        if contig:
            qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
            qlm = np.asarray(qlm, dtype=np.complex128)
        else:
            base = rng.standard_normal(sh.nlm * 2) + 1j * rng.standard_normal(sh.nlm * 2)
            base = np.asarray(base, dtype=np.complex128)
            qlm = base[::2]
            assert not qlm.flags.forc
        inputs.append(qlm)

    originals = [q.copy() for q in inputs]
    sh.synth(*inputs)

    for qlm, original in zip(inputs, originals):
        assert np.array_equal(qlm, original)


@pytest.mark.parametrize("contig", [True, False])
@pytest.mark.parametrize("n_args", [1, 2])
def test_analys_does_not_modify_inputs(contig, n_args):
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(456)

    inputs = []
    for _ in range(n_args):
        base = rng.standard_normal(sh.spat_shape).astype(np.float64)
        if contig:
            v = np.asarray(base, dtype=np.float64)
        else:
            v = base[:, ::-1]
            assert not v.flags.forc
        inputs.append(v)

    originals = [v.copy() for v in inputs]
    sh.analys(*inputs)

    for v, original in zip(inputs, originals):
        assert np.array_equal(v, original)
