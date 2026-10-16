"""Locations and loaders for the declarative parts of the system (robots, worlds, objects, tasks)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOTS_DIR = ROOT / "robots"
WORLDS_DIR = ROOT / "worlds"
OBJECTS_FILE = ROOT / "objects" / "catalog.yaml"
TASKS_DIR = ROOT / "tasks"
BUILD_DIR = ROOT / "build"


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing config file: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _resolve(name_or_path: str, directory: Path, suffix: str) -> Path:
    path = Path(name_or_path)
    if path.suffix:
        return path if path.is_absolute() else (Path.cwd() / path if path.exists() else ROOT / path)
    return directory / f"{name_or_path}{suffix}"


def load_robot(name: str) -> dict:
    spec = load_yaml(_resolve(name, ROBOTS_DIR, ".yaml"))
    for key in ("name", "base", "sensors"):
        if key not in spec:
            raise ConfigError(f"robot spec {name!r} lacks required section {key!r}")
    return spec


def load_world(name: str) -> dict:
    directory = WORLDS_DIR / name
    semantic = load_yaml(directory / "semantic.yaml")
    semantic["dir"] = directory
    semantic["world_path"] = directory / semantic.get("world_file", "world.xml")
    return semantic


def load_catalog() -> dict:
    return load_yaml(OBJECTS_FILE)


def load_task(name_or_path: str) -> dict:
    task = load_yaml(_resolve(name_or_path, TASKS_DIR, ".yaml"))
    task.setdefault("name", Path(name_or_path).stem)
    return task


def list_tasks() -> list[Path]:
    return sorted(TASKS_DIR.glob("*.yaml"))
