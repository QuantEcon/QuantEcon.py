"""
Tests for Notebook Utilities

Functions
---------
fetch_nb_dependencies

"""

from quantecon.util import fetch_nb_dependencies
import os
from unittest import mock

import requests

FILES = ['test_file.md']
REPO = "https://github.com/QuantEcon/QuantEcon.py"
RAW = "raw"
BRANCH = "main"
FOLDER = "quantecon/util/tests/"


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


class TestFetchNbDependenciesErrors:

    def test_http_error_is_not_written_to_disk(self, tmp_path, monkeypatch):
        """
        A failed request must report False and leave no file behind, rather
        than saving the server's error page under the requested filename.
        """
        response = mock.Mock()
        response.content = b"<html>404: Not Found</html>"
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error")

        monkeypatch.chdir(tmp_path)
        with mock.patch("requests.get", return_value=response):
            status = fetch_nb_dependencies(["does-not-exist.csv"], verbose=False)

        assert status == [False]
        assert not os.path.isfile("does-not-exist.csv")

    def test_request_uses_a_timeout(self, tmp_path, monkeypatch):
        """
        Requests must not be able to block indefinitely.
        """
        response = mock.Mock()
        response.content = b"data"
        response.raise_for_status.return_value = None

        monkeypatch.chdir(tmp_path)
        with mock.patch("requests.get", return_value=response) as get:
            fetch_nb_dependencies(["a.csv"], verbose=False, timeout=5)

        assert get.call_args.kwargs["timeout"] == 5

    def test_tuple_of_files_is_accepted(self, tmp_path, monkeypatch):
        """
        A tuple is the natural alternative to a list and must not be treated
        as a directory mapping.
        """
        response = mock.Mock()
        response.content = b"data"
        response.raise_for_status.return_value = None

        monkeypatch.chdir(tmp_path)
        with mock.patch("requests.get", return_value=response):
            status = fetch_nb_dependencies(("a.csv",), verbose=False)

        assert status == [True]
        assert os.path.isfile("a.csv")
