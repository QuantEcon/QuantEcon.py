"""
Tests for Notebook Utilities

Functions
---------
fetch_nb_dependencies

"""

from quantecon.util import fetch_nb_dependencies
from quantecon.util.notebooks import TIMEOUT
import http.server
import os
import socketserver
import threading

import pytest

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


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve one known file and 404 everything else."""

    def do_GET(self):
        if self.path.endswith("present.csv"):
            body, code = b"a,b\n1,2\n", 200
        else:
            body, code = b"<html><title>404 Not Found</title></html>", 404
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:{}".format(server.server_address[1])
    server.shutdown()
    server.server_close()


class TestFetchFailures:
    """A failed request must not be written to disk or reported as success."""

    def test_missing_remote_file_is_not_written(self, local_server, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.warns(UserWarning):
            status = fetch_nb_dependencies(
                files=['absent.csv'], repo=local_server, verbose=False)
        assert status == [False]
        assert not os.path.exists("absent.csv")

    def test_successful_fetch_writes_content(self, local_server, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        status = fetch_nb_dependencies(
            files=['present.csv'], repo=local_server, verbose=False)
        assert status == [True]
        with open("present.csv", "rb") as f:
            assert f.read() == b"a,b\n1,2\n"

    def test_tuple_input_is_accepted(self, local_server, tmp_path, monkeypatch):
        """A tuple must behave like a list rather than falling through to the dict branch."""
        monkeypatch.chdir(tmp_path)
        status = fetch_nb_dependencies(
            files=('present.csv',), repo=local_server, verbose=False)
        assert status == [True]

    def test_timeout_is_forwarded_to_requests(self, local_server, tmp_path, monkeypatch):
        """The request must be bounded so a stalled server cannot block forever."""
        import requests

        seen = {}
        original_get = requests.get

        def spy(url, **kwargs):
            seen.update(kwargs)
            return original_get(url, **kwargs)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(requests, "get", spy)

        #-An explicit timeout is forwarded-#
        fetch_nb_dependencies(
            files=['present.csv'], repo=local_server, verbose=False, timeout=5)
        assert seen.get("timeout") == 5

        #-And the default is a real bound, not None-#
        seen.clear()
        fetch_nb_dependencies(
            files=['present.csv'], repo=local_server, verbose=False, overwrite=True)
        assert seen.get("timeout") == TIMEOUT
        assert TIMEOUT is not None
