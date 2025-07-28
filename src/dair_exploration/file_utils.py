#!/usr/bin/env python3

"""Utility functions for File Operations"""

import os

import git


def repo_dir():
    """Get GIT repo dir when running inside of the git repository"""
    return os.path.normpath(
        git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
    )


def get_config(file: str):
    """Get config from hard-coded config directory"""
    return os.path.join(repo_dir(), os.path.join("config", file))
