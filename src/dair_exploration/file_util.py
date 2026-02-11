#!/usr/bin/env python3

"""Utility functions for File Operations"""
import datetime
from pathlib import Path
import shutil
import pickle
from typing import Any

import gin
import git
import jax


def enable_jax_cache():
    """Enable the filesystem-persistant JIT compilation cache
    See: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
    """
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )


def repo_dir() -> Path:
    """Get GIT repo dir when running inside of the git repository"""
    ret = Path(
        git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )
    assert ret.is_dir()
    return ret


def get_config(file: str) -> Path:
    """Get config from hard-coded config directory"""
    return repo_dir() / "config" / file


@gin.configurable
def results_dir(
    name: str = "run",
    datestr: str = datetime.datetime.now().replace(microsecond=0).isoformat(),
) -> Path:
    """Get config from hard-coded config directory"""
    result_path = repo_dir() / "results" / f"{datestr}_{name}"
    result_path.mkdir(parents=True, exist_ok=True)
    return result_path


def copy_run_config(file_in: Path, out_name: str) -> None:
    """Copy a config file into the run directory"""
    config_results_dir = results_dir() / "config"
    config_results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(file_in, config_results_dir / out_name)


def write_object(obj: Any, subdir: str, out_name: str) -> None:
    """Write an object into the run directory"""
    sub_results_dir = results_dir() / subdir
    sub_results_dir.mkdir(parents=True, exist_ok=True)
    with (sub_results_dir / out_name).open("wb") as file:
        pickle.dump(obj, file)
    print(f"Object written to {(sub_results_dir / out_name).as_posix()}")


def write_text(text: str, subdir: str, out_name: str) -> None:
    """Write an object into the run directory"""
    sub_results_dir = results_dir() / subdir
    sub_results_dir.mkdir(parents=True, exist_ok=True)
    with (sub_results_dir / out_name).open("w") as file:
        file.write(text)
    print(f"Text written to {(sub_results_dir / out_name).as_posix()}")
