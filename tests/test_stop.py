"""Stopping a step partway must leave nothing half written and let it run again."""

import json
import subprocess
import sys
import time

from doppler_loc.gui import server


def test_a_stopped_step_ends_with_its_workers_and_can_run_again(tmp_path, monkeypatch):
    folder = tmp_path / "t"
    (folder / "jobs").mkdir(parents=True)
    (folder / "project.json").write_text('{"settings": {"name": "t", "audio": "a.wav", "date": "2026-01-01", "lat": 0.0, "lon": 0.0}, "passes": [], "adsb": null, "verdicts": {}}')
    monkeypatch.setattr(server, "PROJECTS", tmp_path)
    # A job that would run for a minute, with a child of its own, as a search has its workers.
    code = "import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); open(sys.argv[1], 'w').write(str(c.pid)); time.sleep(60)"
    child_file = tmp_path / "child.txt"
    proc = subprocess.Popen([sys.executable, "-c", code, str(child_file)], start_new_session=sys.platform != "win32")
    server._running[("t", "search")] = proc
    (folder / "jobs" / "search.json").write_text(f'{{"step": "search", "state": "running", "pid": {proc.pid}, "message": "searching", "progress": 0.1}}')
    for _ in range(100):
        if child_file.exists() and child_file.read_text():
            break
        time.sleep(0.1)
    child = int(child_file.read_text())
    assert server._busy("t") == "search"
    status = server.stop_job("t", "search")
    assert status["state"] == "stopped"
    assert proc.poll() is not None
    for _ in range(50):
        if not server._pid_alive(child):
            break
        time.sleep(0.1)
    assert not server._pid_alive(child)
    assert server._busy("t") is None  # so the button works again


def _project_with_flights(tmp_path, monkeypatch):
    region = tmp_path / "region.npz"
    region.write_bytes(b"x")
    folder = tmp_path / "t"
    folder.mkdir()
    state = dict(settings=dict(name="t", audio="a.wav", date="2026-01-01", lat=1.0, lon=2.0, audible_km=10.0), passes=[], verdicts={},
                 adsb=dict(region=str(region), radius_km=60.0, aircraft=5), duration=None)
    (folder / "project.json").write_text(json.dumps(state))
    monkeypatch.setattr(server, "PROJECTS", tmp_path)
    monkeypatch.setattr(server, "get_project", lambda name: None)
    return folder


def test_saving_unchanged_settings_keeps_the_flight_data(tmp_path, monkeypatch):
    folder = _project_with_flights(tmp_path, monkeypatch)
    server.update_settings("t", dict(spacing_m=1000.0, audible_km=10, max_alt_m=6000.0))
    assert json.loads((folder / "project.json").read_text())["adsb"] is not None


def test_changing_the_area_asks_for_the_flight_data_again(tmp_path, monkeypatch):
    folder = _project_with_flights(tmp_path, monkeypatch)
    server.update_settings("t", dict(audible_km=12.0))
    assert json.loads((folder / "project.json").read_text())["adsb"] is None

