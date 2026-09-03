from pathlib import Path
import importlib.util


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "check_tr3d_environment.py"
)
SPEC = importlib.util.spec_from_file_location("check_tr3d_environment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rc_version_ranges():
    assert MODULE._in_range("2.0.0rc4", "2.0.0rc4", "2.2.0")
    assert MODULE._in_range("2.1.0", "2.0.0rc4", "2.2.0")
    assert not MODULE._in_range("1.7.2", "2.0.0rc4", "2.2.0")
    assert not MODULE._in_range("2.2.0", "2.0.0rc4", "2.2.0")


def test_locked_vendor_files_exist():
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party" / "mmdetection3d"
    assert (vendor / "projects" / "TR3D" / "tr3d" / "tr3d_head.py").is_file()
    assert (
        vendor
        / "projects"
        / "TR3D"
        / "configs"
        / "tr3d_1xb16_scannet-3d-18class.py"
    ).is_file()


def test_class_agnostic_head_and_abi_contract_are_explicit():
    assert MODULE.SUPPORTED_HEADS == {
        "TR3DHead",
        "TR3DClassAgnosticHead",
    }
    assert MODULE.ABI_PINS == {
        "numpy": "1.23.5",
        "numba": "0.56.4",
        "llvmlite": "0.39.1",
    }
