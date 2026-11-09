import numpy as np

from doppler_loc import scenarios
from doppler_loc.classify import FEATURES, PassModel, pass_features
from doppler_loc.detect import detect, measure
from doppler_loc.spectro import compute_spectrogram


def test_a_clean_pass_looks_like_one():
    sc = scenarios.two_passes(n_decoys=0)
    spec = compute_spectrogram(sc.audio, sc.sr)
    det = detect(spec, max_events=5)[0]
    ev = measure(spec, det)
    rec = dict(line_t=ev.line[0], line_f=ev.line[1], f0=ev.f0, speed=ev.speed, dist=ev.dist, t_heard=ev.t_heard,
               strength=det.strength, harmonics=det.harmonics)
    f = pass_features(spec, rec)
    assert set(f) == set(FEATURES)
    assert all(np.isfinite(v) for v in f.values())
    assert f["fit_share"] > 0.8
    assert f["contrast_db"] > 6.0
    assert f["visible"] > 0.8
    assert f["stripes"] < 0.1
    assert f["harmonics"] >= 1


def test_logistic_model_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    rows = [dict(a=rng.normal(), b=rng.normal()) for _ in range(300)]
    y = np.array([r["a"] - 0.5 * r["b"] + 0.3 * rng.normal() > 1.0 for r in rows], float)
    model = PassModel.train(rows, y, names=("a", "b"))
    assert model.weights[1] > 0 > model.weights[2]
    p = model.prob(rows)
    assert np.mean((p > 0.5) == (y > 0)) > 0.85
    model.save(tmp_path / "m.json")
    assert np.allclose(PassModel.load(tmp_path / "m.json").prob(rows), p)
