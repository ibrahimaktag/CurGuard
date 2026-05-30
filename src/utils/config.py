"""Configuration loader and environment-aware path resolver.

Supports two execution environments transparently:

* **Local** (development / testing)
  - Project root is resolved via ``Path(__file__)``.
  - Raw dataset paths are relative to the project root.
  - Artifacts are written under ``<project_root>/artifacts/``.

* **Kaggle** (cloud GPU training)
  - Detected via the ``KAGGLE_KERNEL_RUN_TYPE`` environment variable that
    Kaggle injects into every kernel session automatically.
  - Raw datasets live under read-only ``/kaggle/input/`` mounts.
  - Artifacts are written to the writable ``/kaggle/working/artifacts/``.

Public API
----------
is_kaggle()                          → bool
get_project_root()                   → Path
load_config(name)                    → dict
get_config_path(name)                → Path
get_dataset_path(domain, cfg?)       → Path   (raw CSV directory)
get_artifacts_root(cfg?)             → Path   (write-able output root)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel session.

    Kaggle automatically sets ``KAGGLE_KERNEL_RUN_TYPE`` to a non-empty
    string (e.g. ``'Interactive'``, ``'Batch'``) for every kernel.  A local
    machine will not have this variable unless the user sets it manually.

    Returns:
        bool: True if the code is executing inside a Kaggle kernel.
    """
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    """Compute the absolute project root path.

    Local:  ``src/utils/config.py`` is three levels below the repo root,
            so ``Path(__file__).parents[2]`` gives the correct root.
    Kaggle: ``/kaggle/working`` is the writable working directory and the
            conventional location where the repo is cloned or extracted.
            We still resolve via ``__file__`` because Kaggle executes the
            module from the same relative position inside the repo.

    Returns:
        Path: Absolute project root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _load_yaml(config_path: str) -> dict[str, Any]:
    """Load and cache a YAML file.  Internal use only.

    Args:
        config_path: Absolute path string to the YAML file.

    Returns:
        Parsed dict.
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_name: str) -> dict[str, Any]:
    """Load a YAML configuration file from configs/.

    Results are cached per config name so repeated calls in the same
    process incur no disk I/O.

    Args:
        config_name: Filename without extension (e.g. ``'datasets'``,
                     ``'cloud_model'``).

    Returns:
        Parsed dict from the YAML file.

    Raises:
        FileNotFoundError: Config file does not exist.
        yaml.YAMLError: YAML parsing fails.
    """
    root = _resolve_project_root()
    config_path = root / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return _load_yaml(str(config_path))


def get_config_path(config_name: str) -> Path:
    """Return the absolute path to a named config file.

    Args:
        config_name: Filename without extension.

    Returns:
        Path: Absolute path to the ``.yaml`` file.
    """
    return _resolve_project_root() / "configs" / f"{config_name}.yaml"


# ---------------------------------------------------------------------------
# Public path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root directory.

    On Kaggle the project root is the repo directory inside
    ``/kaggle/working/`` (e.g. ``/kaggle/working/C-rGuard``).
    The path is computed from ``__file__`` so it is correct in both
    environments without any hard-coded strings.

    Returns:
        Path: Absolute project root.
    """
    return _resolve_project_root()


def get_dataset_path(domain: str, cfg: dict[str, Any] | None = None) -> Path:
    """Resolve the raw-data directory for a domain.

    Selects between ``raw_path`` (local) and ``kaggle_raw_path`` (Kaggle)
    from ``datasets.yaml`` automatically based on :func:`is_kaggle`.

    Args:
        domain:  Dataset key as defined in ``datasets.yaml``
                 (``'cloud'`` or ``'iot'``).
        cfg:     Pre-loaded datasets config dict.  If ``None``, the file
                 is loaded automatically.  Pass an already-loaded dict to
                 avoid redundant I/O inside tight loops.

    Returns:
        Path: Absolute path to the raw CSV directory.

    Raises:
        KeyError:  ``domain`` not present in config.
        KeyError:  Expected path key missing in config section.
        FileNotFoundError: Resolved path does not exist on the filesystem.
    """
    if cfg is None:
        cfg = load_config("datasets")

    if domain not in cfg:
        raise KeyError(
            f"Domain '{domain}' not found in datasets.yaml. "
            f"Available: {list(cfg.keys())}"
        )

    domain_cfg = cfg[domain]

    if is_kaggle():
        path_key = "kaggle_raw_path"
        raw = Path(domain_cfg[path_key])          # already absolute on Kaggle
    else:
        path_key = "raw_path"
        raw = _resolve_project_root() / domain_cfg[path_key]

    if not raw.exists():
        raise FileNotFoundError(
            f"Raw data directory for domain '{domain}' not found: {raw}\n"
            f"  (resolved via '{path_key}' in datasets.yaml, "
            f"environment={'kaggle' if is_kaggle() else 'local'})"
        )

    return raw


def get_artifacts_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return the write-able artifacts root directory for the current environment.

    Creates the directory (including parents) if it does not yet exist.

    Environment mapping
    -------------------
    Local  → ``<project_root>/artifacts/``
    Kaggle → ``/kaggle/working/artifacts/``

    Both paths come from the ``environment`` section of ``datasets.yaml``
    so they can be customised without touching Python code.

    Args:
        cfg: Pre-loaded datasets config dict.  Loaded automatically when
             ``None``.

    Returns:
        Path: Absolute, guaranteed-to-exist artifacts root directory.
    """
    if cfg is None:
        cfg = load_config("datasets")

    env_key = "kaggle" if is_kaggle() else "local"

    try:
        raw_root = cfg["environment"][env_key]["artifacts_root"]
    except KeyError as exc:
        raise KeyError(
            f"Missing 'environment.{env_key}.artifacts_root' in datasets.yaml"
        ) from exc

    if is_kaggle():
        root = Path(raw_root)              # absolute path string from YAML
    else:
        root = _resolve_project_root() / raw_root   # relative → absolute

    root.mkdir(parents=True, exist_ok=True)
    return root
