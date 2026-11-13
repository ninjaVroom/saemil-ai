"""CPU reporting must work where the Windows registry module is unavailable."""
import sys
from types import SimpleNamespace

from scripts import benchmark


def test_cpu_name_without_windows_registry(monkeypatch):
    monkeypatch.setattr(benchmark, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setitem(sys.modules, "winreg", None)
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "Test CPU")
    assert benchmark.cpu_name() == "Test CPU"
