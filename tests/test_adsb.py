import json
import urllib.request

from doppler_loc.adsb import download_day


def test_a_day_already_downloaded_is_not_asked_about_again(tmp_path, monkeypatch):
    sizes = {"v2026.11.04-planes-readsb-prod-0.tar.aa": 5, "v2026.11.04-planes-readsb-prod-0.tar.ab": 3}
    for name, size in sizes.items():
        (tmp_path / f"prod.tar.{name.rsplit('.', 1)[1]}").write_bytes(b"x" * size)
    (tmp_path / "assets.json").write_text(json.dumps(sizes))

    def refuse(*_, **__):
        raise AssertionError("asked GitHub about a day that is all there")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert [p.name for p in download_day("2026-11-04", tmp_path)] == ["prod.tar.aa", "prod.tar.ab"]
