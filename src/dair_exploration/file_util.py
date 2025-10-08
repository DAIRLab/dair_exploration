#!/usr/bin/env python3

"""Utility functions for File Operations"""
import os
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


def repo_dir():
    """Get GIT repo dir when running inside of the git repository"""
    return os.path.normpath(
        git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )


def get_config(file: str):
    """Get config from hard-coded config directory"""
    return os.path.join(repo_dir(), os.path.join("config", file))
