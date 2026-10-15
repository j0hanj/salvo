from dataclasses import dataclass, field

import pytest

from configs.load import dump, from_dict, load


@dataclass
class Inner:
    a: int = 1


@dataclass
class Outer:
    name: str = "x"
    inner: Inner = field(default_factory=Inner)


def test_nested():
    cfg = from_dict(Outer, {"name": "y", "inner": {"a": 5}})
    assert cfg == Outer("y", Inner(5))


def test_unknown_key_fails():
    with pytest.raises(KeyError):
        from_dict(Outer, {"nmae": "typo"})


def test_roundtrip(tmp_path):
    cfg = Outer("z", Inner(3))
    dump(cfg, tmp_path / "c.yaml")
    assert load(Outer, tmp_path / "c.yaml") == cfg
