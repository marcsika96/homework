import hashlib
import json
from collections import Counter
from pathlib import Path

from reviewgraph.schemas import Unit

DATA = Path(__file__).resolve().parent.parent / "data"

# The test set was frozen on 2026-11-12, before the first test run.
# If this checksum changes, the evaluation in EVALUATION.md no longer describes this file.
TEST_SET_SHA256 = "50e9c9dfd58b5f3c577414a4def7d423d4ae634db19e2563384703043526d5de"


def load_units(name: str) -> list[Unit]:
    lines = (DATA / name).read_text(encoding="utf-8").splitlines()
    return [Unit.model_validate_json(line) for line in lines]


def test_test_set_is_frozen():
    content = (DATA / "units_test.jsonl").read_bytes()

    assert hashlib.sha256(content).hexdigest() == TEST_SET_SHA256


def test_unit_counts_match_the_specification():
    dev_counts = Counter(unit.increment for unit in load_units("units_dev.jsonl"))
    test_counts = Counter(unit.increment for unit in load_units("units_test.jsonl"))

    assert dev_counts == {"I1": 9, "I2": 8, "I3": 9}
    assert test_counts == {"I4": 6, "I5": 10, "I6": 3, "I7": 7, "I8": 10}


def test_splits_are_labelled_and_do_not_overlap():
    dev_units = load_units("units_dev.jsonl")
    test_units = load_units("units_test.jsonl")

    assert all(unit.split == "dev" for unit in dev_units)
    assert all(unit.split == "test" for unit in test_units)
    assert not {unit.commit for unit in dev_units} & {unit.commit for unit in test_units}


def test_units_point_to_the_recorded_increment_commits():
    increments = json.loads((DATA / "increments.json").read_text(encoding="utf-8"))

    for unit in load_units("units_dev.jsonl") + load_units("units_test.jsonl"):
        assert unit.commit == increments[unit.increment]
        assert unit.id == f"{unit.increment}:{unit.path}"
