"""yaml -> dataclass. strict on purpose: a typo'd key should blow up, not get silently ignored."""

import dataclasses
import typing
from pathlib import Path

import yaml


def from_dict(cls, data: dict):
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} isn't a dataclass")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        t = hints[name]
        # recurse into nested dataclasses, everything else passes through as-is
        kwargs[name] = from_dict(t, value) if dataclasses.is_dataclass(t) else value
    return cls(**kwargs)


def load(cls, path: str | Path):
    with open(path) as f:
        return from_dict(cls, yaml.safe_load(f) or {})


def dump(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f, sort_keys=False)
