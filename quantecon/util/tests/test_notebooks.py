"""
Tests for Notebook Utilities

Functions
---------
fetch_nb_dependencies

"""

from quantecon.util import fetch_nb_dependencies
import os
import pytest

FILES = ['test_file.md']
REPO = "https://github.com/QuantEcon/QuantEcon.py"
RAW = "raw"
BRANCH = "main"
FOLDER = "quantecon/util/tests/"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestNotebookUtils:

    def test_fetch_nb_dependencies(self):
        """
        Run First and Test Download
        """
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)
        assert(not (False in status))

    def test_fetch_nb_dependencies_overwrite(self):
        """
        Run Second and Ensure file is skipped by checking a False is found in status
        """
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)  #First will succeed
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)  #Second should skip
        assert(False in status)

    def teardown_method(self):
        os.remove("test_file.md")


class TestNotebookUtilsDeprecation:
    """`fetch_nb_dependencies` is deprecated in 0.12.0 and removed in v1.0."""

    def test_fetch_nb_dependencies_warns(self):
        with pytest.warns(DeprecationWarning, match="removed in v1.0"):
            fetch_nb_dependencies(
                files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)

    def teardown_method(self):
        if os.path.isfile("test_file.md"):
            os.remove("test_file.md")
