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
        kwargs[name] = _convert(hints[name], value)
    return cls(**kwargs)


def _convert(t, value):
    # recurse into nested dataclasses and list[dataclass], everything else passes through as-is
    if dataclasses.is_dataclass(t):
        return from_dict(t, value)
    args = typing.get_args(t)
    if typing.get_origin(t) is list and args and dataclasses.is_dataclass(args[0]):
        return [from_dict(args[0], v) for v in value]
    return value


def load(cls, path: str | Path):
    with open(path) as f:
        return from_dict(cls, yaml.safe_load(f) or {})


def dump(obj, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f, sort_keys=False)
