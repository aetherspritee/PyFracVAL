"""Unit tests for pyfracval.dask_runner -- specifically the
install_wheel_on_workers()/_register_package() wheel-shipping mechanism.

Deliberately does NOT spin up a real LocalCluster or run a real `pip
install` -- the embedded install/fingerprint functions do real subprocess
package installs into whatever Python environment the worker process
shares (for a LocalCluster on this machine, that's this same devenv venv),
which would be an invasive, slow side effect for a unit test to have. A
fake client that never actually calls the passed function is enough to
verify the dispatch/error-handling logic that this module's refactor
actually changed.
"""

from __future__ import annotations

import pytest

from pyfracval.dask_runner import install_wheel_on_workers


class _FakeClient:
    """Records what would be sent to the scheduler/workers without
    actually invoking it -- install_wheel_on_workers()'s embedded
    functions do real subprocess pip installs, which this must not
    trigger.

    Notably: register_plugin() is recorded but its plugin's setup() is
    deliberately never called (unlike the real Client, which calls it on
    every worker) -- that's exactly the real subprocess-install path this
    fake exists to avoid triggering.
    """

    def __init__(self, worker_ids, fingerprint_version):
        self._worker_ids = worker_ids
        self._fingerprint_version = fingerprint_version
        self.scheduler_calls: list[tuple] = []
        self.run_calls: list[tuple] = []
        self.register_plugin_calls: list[tuple] = []

    def run_on_scheduler(self, fn, *args):
        self.scheduler_calls.append((fn, args))
        return {"installed_wheel": "x"}

    def scheduler_info(self):
        return {"workers": {wid: {} for wid in self._worker_ids}}

    def register_plugin(self, plugin, name=None):
        self.register_plugin_calls.append((plugin, name))
        return {w: {"status": "OK"} for w in self._worker_ids}

    def run(self, fn, *args, workers=None):
        self.run_calls.append((fn, args, workers))
        # Only the fingerprint call goes through client.run() now --
        # installation happens via register_plugin() above.
        return {
            w: {
                "version": self._fingerprint_version,
                "module_file": "fake.py",
                "pid": "1",
            }
            for w in workers
        }


def _wheel_file(tmp_path, name="dummy-1.2.3-py3-none-any.whl"):
    path = tmp_path / name
    path.write_bytes(b"not a real wheel, never actually installed")
    return path


def test_install_wheel_on_workers_dispatches_with_correct_args(tmp_path):
    wheel_path = _wheel_file(tmp_path)
    client = _FakeClient(
        worker_ids=["tcp://w1", "tcp://w2"], fingerprint_version="1.2.3"
    )

    install_wheel_on_workers(client, wheel_path, "dummy", "1.2.3")

    # Scheduler install call.
    assert len(client.scheduler_calls) == 1
    _, sched_args = client.scheduler_calls[0]
    (
        wheel_bytes,
        wheel_filename,
        package_name,
        expected_version,
        env_wheel,
        env_version,
    ) = sched_args
    assert wheel_bytes == wheel_path.read_bytes()
    assert wheel_filename == wheel_path.name
    assert package_name == "dummy"
    assert expected_version == "1.2.3"
    # Generalized env var names -- must match what environments.py's own
    # PYFRACVAL_INSTALLED_WHEEL/PYFRACVAL_EXPECTED_VERSION constants equal
    # for the package_name="pyfracval" case (verified separately below).
    assert env_wheel == "DUMMY_INSTALLED_WHEEL"
    assert env_version == "DUMMY_EXPECTED_VERSION"

    # Installation now happens via a registered WorkerPlugin, not a
    # client.run() sweep -- covers workers that join later too.
    assert len(client.register_plugin_calls) == 1
    plugin, plugin_name = client.register_plugin_calls[0]
    assert plugin.wheel_bytes == wheel_path.read_bytes()
    assert plugin.wheel_filename == wheel_path.name
    assert plugin.package_name == "dummy"
    assert plugin.expected_version == "1.2.3"
    assert plugin.env_installed_wheel == "DUMMY_INSTALLED_WHEEL"
    assert plugin.env_expected_version == "DUMMY_EXPECTED_VERSION"
    assert plugin_name == plugin.name

    # One client.run() call: the post-install fingerprint check.
    assert len(client.run_calls) == 1
    (fingerprint_call,) = client.run_calls
    assert fingerprint_call[1] == ("dummy",)  # args=(package_name,)
    assert fingerprint_call[2] == ["tcp://w1", "tcp://w2"]


def test_install_wheel_on_workers_env_var_names_match_pyfracval_convention():
    """package_name="pyfracval" must reconstruct exactly the same env var
    names environments.py's own PYFRACVAL_INSTALLED_WHEEL/
    PYFRACVAL_EXPECTED_VERSION constants hold -- this refactor stopped
    importing those constants directly in favor of deriving the same
    strings from package_name, so pin they still agree."""
    from pyfracval.environments import (
        PYFRACVAL_EXPECTED_VERSION,
        PYFRACVAL_INSTALLED_WHEEL,
    )

    assert f"{'pyfracval'.upper()}_INSTALLED_WHEEL" == PYFRACVAL_INSTALLED_WHEEL
    assert f"{'pyfracval'.upper()}_EXPECTED_VERSION" == PYFRACVAL_EXPECTED_VERSION


def test_install_wheel_on_workers_raises_on_version_mismatch(tmp_path):
    wheel_path = _wheel_file(tmp_path)
    client = _FakeClient(worker_ids=["tcp://w1"], fingerprint_version="0.0.1-stale")

    with pytest.raises(RuntimeError, match="version mismatch"):
        install_wheel_on_workers(client, wheel_path, "dummy", "1.2.3")


def test_register_package_delegates_to_install_wheel_on_workers(tmp_path, monkeypatch):
    """_register_package() is now a thin wrapper: build pyfracval's own
    wheel, then call the generic install_wheel_on_workers() with
    package_name="pyfracval". Pin that delegation instead of re-testing
    install_wheel_on_workers()'s own behavior (covered above)."""
    import pyfracval.dask_runner as dask_runner_mod

    fake_wheel = _wheel_file(tmp_path, "pyfracval-9.9.9-py3-none-any.whl")
    monkeypatch.setattr(dask_runner_mod, "_build_wheel", lambda: fake_wheel)
    monkeypatch.setattr(dask_runner_mod, "_project_version", lambda: "9.9.9")

    seen = {}

    def spy_install(client, wheel_path, package_name, expected_version):
        seen.update(
            client=client,
            wheel_path=wheel_path,
            package_name=package_name,
            expected_version=expected_version,
        )

    monkeypatch.setattr(dask_runner_mod, "install_wheel_on_workers", spy_install)

    fake_client = object()
    dask_runner_mod._register_package(fake_client)

    assert seen["client"] is fake_client
    assert seen["wheel_path"] == fake_wheel
    assert seen["package_name"] == "pyfracval"
    assert seen["expected_version"] == "9.9.9"
