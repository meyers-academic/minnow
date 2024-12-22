import discovery as ds
import minnow as mn
from pathlib import Path

def test_load_pulsar():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    psr_files = [
            data_dir / "v1p1_de440_pint_bipm2019-B1855+09.feather",
            data_dir / "v1p1_de440_pint_bipm2019-B1953+29.feather",
        ]
    psr = mn.MultiBandPulsar.read_feather_pre_process(psr_files[0])
    print(len(psr.residuals))
    assert len(psr.residuals) == 360
    assert isinstance(psr, mn.MultiBandPulsar)