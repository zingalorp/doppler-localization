import numpy as np

from doppler_loc import scenarios
from doppler_loc.detect import assess, close_to_found, detect, harmonic_groups, measure
from doppler_loc.measure import Event, fit_flyby, straight_flyby, trace_ridge, turning_flyby
from doppler_loc.physics import closest_approach
from doppler_loc.score import line_agreement
from doppler_loc.spectro import compute_spectrogram


def test_two_passes_found_and_measured_without_help():
    sc = scenarios.two_passes(n_decoys=0)
    spec = compute_spectrogram(sc.audio, sc.sr)
    dets = detect(spec, max_events=5)
    found = sorted(dets[:2], key=lambda d: d.t_heard)
    for k, (det, mark, track) in enumerate(zip(found, sc.marks, sc.true_tracks)):
        true_heard = closest_approach(track, sc.truth[None]).t_heard[0] - sc.t_rec
        assert abs(det.t_heard - true_heard) < 6.0
        assert 2 in det.harmonics
        ev = measure(spec, det)
        boxed = fit_flyby(*trace_ridge(spec, *mark))  # traced inside a hand-drawn box
        assert abs(ev.t_heard - boxed.t_heard) < 2.0
        assert abs(ev.f0 / boxed.f0 - 1) < 0.01
        # The aircraft that made it, at the true spot and time, follows the traced line.
        assert line_agreement(track, sc.truth[None], [sc.t_rec], [sc.f0s[k]], ev)[0] > 0.8
    assert all(d.strength < 0.5 * found[0].strength for d in dets[2:])


def test_steady_and_stepping_tones_are_not_passes():
    sr, dur = 8000, 600.0
    rng = np.random.default_rng(1)
    t = np.arange(int(sr * dur)) / sr
    audio = rng.normal(0, 1, t.size)
    audio += 0.3 * np.sin(2 * np.pi * 150.0 * t)  # steady hum
    step = np.where((t // 25) % 2 == 0, 80.0, 70.0)  # machinery switching between two speeds
    audio += 0.3 * np.sin(2 * np.pi * np.cumsum(step) / sr)
    for start in rng.uniform(0, dur - 5, 12):  # broadband bursts
        on = (t > start) & (t < start + 3)
        audio[on] += rng.normal(0, 4, on.sum())
    spec = compute_spectrogram(audio, sr)
    assert all(d.strength < 20.0 for d in detect(spec))


def test_passes_are_classified_for_review():
    t = np.linspace(-60.0, 60.0, 241)
    clean = straight_flyby(t, 100.0, 60.0, 2000.0, 0.0)
    ev = Event(t_heard=0.0, f0=100.0, speed=60.0, dist=2000.0, line=(t, clean))
    assert assess(ev, strength=45.0).kind == "likely aircraft"
    assert assess(ev, strength=12.0).kind == "possible"  # faint
    assert assess(ev, strength=20.0, harmonics=[2]).confidence > assess(ev, strength=20.0).confidence
    # A strong line whose pitch wobbles, as a banking plane's does: no straight pass follows it, but
    # it is kept for a person to judge rather than rejected.
    turning = assess(fit_flyby(t, clean + 3.0 * np.sin(2 * np.pi * t / 40)), strength=40.0)
    assert turning.kind == "possibly turning" and turning.dist_tol > 0.7
    # A plane turning towards the microphone rises in pitch before falling, which no straight pass
    # does: a turning flight follows it, and is what gets drawn.
    bend = turning_flyby(t, 120.0, 60.0, 0.04, 1500.0, -2.5, 500.0, 0.0)
    ev = fit_flyby(t, bend)
    turned = assess(ev, strength=40.0)
    straight_share = np.mean(np.abs(straight_flyby(t, ev.f0, ev.speed, ev.dist, ev.t_heard) - bend) < 1.5)
    assert np.argmax(bend) > 10 and straight_share < 0.75
    assert turned.turn is not None and turned.turn.share > 0.9 and turned.kind == "possibly turning"
    assert abs(turned.turn.params[0] - 120.0) < 2.0
    far = Event(t_heard=0.0, f0=100.0, speed=60.0, dist=15000.0, line=(t, straight_flyby(t, 100.0, 60.0, 15000.0, 0.0)))
    assert assess(far, strength=45.0).kind == "not a pass"


def test_harmonics_of_one_aircraft_are_grouped():
    t = np.arange(-60.0, 60.0, 0.5)
    pitch = straight_flyby(t, 1.0, 60.0, 1500.0, 0.0)
    rng = np.random.default_rng(0)
    wobble = lambda n=t.size: rng.normal(0, 0.15, n)  # half a frequency bin of tracing noise, in Hz
    lines = [
        (t, 100.0 * pitch + wobble()),  # 0: one aircraft...
        (t[40:], 150.0 * pitch[40:] + wobble(t.size - 40)),  # 1: ...its 3/2 line, traced over part of the pass
        (t, 133.0 * straight_flyby(t, 1.0, 90.0, 4000.0, 8.0) + wobble()),  # 2: another aircraft, same time
        (t, np.full(t.size, 60.0)), (t, np.full(t.size, 120.0)),  # 3, 4: hum and its harmonic
    ]
    assert harmonic_groups(lines) == [[0, 1]]


def _falling(t, f_start, f_end, mid, width=12.0):
    return f_end + (f_start - f_end) * (1 - np.tanh((t - mid) / width)) / 2


def test_two_planes_that_converge_are_not_one_line():
    t = np.arange(0.0, 120.0, 0.5)
    a = _falling(t, 102.0, 78.0, 40.0)
    b = a * (1 + 0.1 * np.clip(1 - t / 80.0, 0.0, 1.0))  # apart, closing, then together
    assert harmonic_groups([(t, a), (t, b)]) == []


def test_the_same_line_traced_twice_is_still_one():
    t = np.arange(0.0, 120.0, 0.5)
    a = _falling(t, 102.0, 78.0, 60.0)
    noise = np.random.default_rng(0).normal(0.0, 0.25, t.size)
    assert harmonic_groups([(t, a), (t, a + noise)]) == [[0, 1]]


def test_close_harmonics_are_still_one_plane():
    t = np.arange(0.0, 120.0, 0.5)
    a = _falling(t, 102.0, 78.0, 60.0)
    assert harmonic_groups([(t, a), (t, a * 9 / 8)]) == [[0, 1]]
    assert harmonic_groups([(t, a), (t, a * 1.5)]) == [[0, 1]]


def _flybys(*planes, sr=2000, dur=600.0, seed=3):
    t = np.arange(int(sr * dur)) / sr
    audio = np.random.default_rng(seed).normal(0.0, 1.0, t.size)
    for f0, speed, dist, t_heard, level in planes:
        f = straight_flyby(t, f0, speed, dist, t_heard)
        audio += level * np.sin(2 * np.pi * np.cumsum(f) / sr)
    return compute_spectrogram(audio, sr)


def test_a_curve_hidden_beside_a_louder_one_is_found():
    spec = _flybys((100.0, 60.0, 1500.0, 300.0, 0.5), (96.0, 85.0, 2500.0, 310.0, 0.25))
    first = [measure(spec, d) for d in detect(spec, max_events=15) if d.strength > 28.0]
    assert len(first) == 1  # the second lies too close for detection itself
    found = close_to_found(spec, [e.line for e in first])
    assert len(found) == 1
    t, f = found[0][1].line
    assert np.median(np.abs(f - straight_flyby(t, 96.0, 85.0, 2500.0, 310.0))) < 1.0


def test_what_a_lone_curve_leaves_behind_is_not_a_curve():
    spec = _flybys((100.0, 60.0, 1500.0, 300.0, 0.5))
    first = [measure(spec, d) for d in detect(spec, max_events=15) if d.strength > 28.0]
    assert len(first) == 1
    assert close_to_found(spec, [e.line for e in first]) == []
