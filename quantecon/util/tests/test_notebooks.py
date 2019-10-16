"""
Tests for Notebook Utilities

Functions
---------
fetch_nb_dependencies

"""

from quantecon.util import fetch_nb_dependencies
import unittest
import os

FILES = ['test_file.md']
REPO = "https://github.com/QuantEcon/QuantEcon.py"
RAW = "raw"
BRANCH = "master"
FOLDER = "quantecon/util/tests/"

class TestNotebookUtils(unittest.TestCase):

    def test_fetch_nb_dependencies(self):
        """
        Run First and Test Download
        """
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)
        self.assertFalse(False in status)

    def test_fetch_nb_dependencies_overwrite(self):
        """
        Run Second and Ensure file is skipped by checking a False is found in status
        """
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)  #First will succeed
        status = fetch_nb_dependencies(
            files=FILES, repo=REPO, raw=RAW, branch=BRANCH, folder=FOLDER)  #Second should skip
        self.assertTrue(False in status)

    def tearDown(self):
        os.remove("test_file.md")
