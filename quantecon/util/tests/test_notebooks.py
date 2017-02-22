"""
Tests for Notebook Utilities

Functions
---------
fetch_nb_dependencies

"""

from quantecon.util import fetch_nb_dependencies
import unittest

FILES = ['README.md']
REPO = "https://github.com/QuantEcon/QuantEcon.py"
RAW = "raw"
BRANCH = "master"

class TestNotebookUtils(unittest.TestCase):

    def test_fetch_nb_dependencies(self):
        """
        Run First and Test Download
        """
        status = fetch_nb_dependencies(files=FILES, repo=REPO, raw=RAW, branch=BRANCH)
        self.assertFalse(False in status)

    def test_fetch_nb_dependencies_overwrite(self):
        """
        Run Second and Ensure file is skipped by checking a False is found in status
        """
        status = fetch_nb_dependencies(files=FILES, repo=REPO, raw=RAW, branch=BRANCH)
        self.assertTrue(False in status)