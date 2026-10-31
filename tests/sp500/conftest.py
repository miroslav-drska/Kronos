"""Unit tests cannot accidentally contact public services."""
import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, request):
    if request.node.get_closest_marker("slow"):
        return
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in unit tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr("sp500.data.yf.download", blocked)
    monkeypatch.setattr("requests.sessions.Session.request", blocked)