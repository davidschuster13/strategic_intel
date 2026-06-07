import json
from pathlib import Path
from typing import Dict, List

from .policy_mechanics import merge_side_records

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMETER_DIR = str(_PACKAGE_DIR / "parameterized_output")
_DEFAULT_MODELS_DIR = str(_PACKAGE_DIR / "models" / "v8_balanced")
_CACHE: Dict[str, object] = {}


def default_models_dir() -> str:
    return _DEFAULT_MODELS_DIR


def clear_parameter_cache():
    _CACHE.clear()


def load_all_parameters(json_dir: str = DEFAULT_PARAMETER_DIR, quiet: bool = False, use_cache: bool = True):
    """Load merged, side-specific policy parameters for Blue and Red agents."""
    cache_key = str(Path(json_dir).resolve())
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    parameters: Dict[str, object] = {}

    root = Path(json_dir)

    def _read_json(path: Path):
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _records_for_side(side: str) -> List[dict]:
        side_dir = root / side
        master = _read_json(side_dir / "all_parameterized_policies.json")
        if isinstance(master, list) and master:
            return master
        records: List[dict] = []
        if side_dir.is_dir():
            for path in sorted(side_dir.glob("params_*.json")):
                data = _read_json(path)
                if isinstance(data, dict):
                    records.append(data)
        return records

    blue_records = _records_for_side("blue")
    red_records = _records_for_side("red")

    if not blue_records and not red_records:
        master = _read_json(root / "all_parameterized_policies.json")
        if isinstance(master, dict):
            blue_records = master.get("blue", [])
            red_records = master.get("red", [])
        elif isinstance(master, list):
            blue_records = master

    parameters["Blue_US"] = merge_side_records(blue_records)
    parameters["Red_Adversary"] = merge_side_records(red_records)
    parameters["Blue_US_records"] = blue_records
    parameters["Red_Adversary_records"] = red_records

    blue_n = parameters["Blue_US"].get("n_source_documents", 0) if parameters["Blue_US"] else 0
    red_n = parameters["Red_Adversary"].get("n_source_documents", 0) if parameters["Red_Adversary"] else 0
    if not quiet:
        print(f"✅ Loaded agents: Blue ({blue_n} docs merged), Red ({red_n} docs merged)")

    if use_cache:
        _CACHE[cache_key] = parameters
    return parameters
