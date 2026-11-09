import numpy as np

from doppler_loc.flight import Maneuver, adsb_sample, simulate_track
from doppler_loc.measure import Event, fit_flyby, implausible, straight_flyby, trace_ridge
from doppler_loc.physics import closest_approach, doppler_factor
from doppler_loc.spectro import compute_spectrogram
from doppler_loc.synth import Source, synthesize

C = 343.0


def straight_track(speed=70.0, alt=1000.0, offset=800.0):
    # Due north along x = offset, passing (offset, 0) at t = 1000.
    return simulate_track("T", (offset, -speed * 300, alt), 0.0, speed, 700.0, 600.0)


def test_doppler_factor_matches_closed_form():
    tr = straight_track()
    obs = np.array([0.0, 0.0, 0.0])
    dist = np.hypot(800.0, 1000.0)
    t = np.linspace(900, 1100, 41)
    got = 100.0 * doppler_factor(tr, obs, t, C)
    want = straight_flyby(t, 100.0, 70.0, dist, 1000.0 + dist / C, C)
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_retarded_curve_is_not_a_delayed_instantaneous_curve():
    tr = straight_track()
    obs = np.array([0.0, 0.0, 0.0])
    delay = np.hypot(800.0, 1000.0) / C
    t = np.linspace(950, 1050, 101)
    retarded = doppler_factor(tr, obs, t, C)
    delayed = doppler_factor(tr, obs, t - delay, C, retarded=False)
    # Same value at the CPA sound, but the shapes differ away from it.
    assert np.nanmax(np.abs(retarded - delayed)) * 100 > 0.5


def test_closest_approach_straight_line():
    tr = straight_track()
    obs = np.array([[0.0, 0.0, 0.0], [0.0, 2000.0, 200.0]])
    ca = closest_approach(tr, obs, C)
    np.testing.assert_allclose(ca.t, [1000.0, 1000.0 + 2000 / 70], atol=1e-3)
    np.testing.assert_allclose(ca.dist, [np.hypot(800, 1000), np.hypot(800, 800)], rtol=1e-6)
    np.testing.assert_allclose(ca.t_heard, ca.t + ca.dist / C)
    assert ca.valid.all()


def test_adsb_track_keeps_velocity_accurate():
    tr = simulate_track("T", (0, 0, 1000), 45.0, 65.0, 0.0, 600.0, [Maneuver(200, 230, turn_rate=3.0)])
    ad = adsb_sample(tr, rng=1)
    t = np.linspace(20, 580, 200)
    err = np.linalg.norm(ad.velocity(t) - tr.velocity(t), axis=1)
    assert np.median(err) < 0.5


def test_fit_flyby_recovers_parameters():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 120.0, 0.5)
    f = straight_flyby(t, 90.0, 60.0, 1500.0, 61.0, C) + rng.normal(0, 0.2, t.size)
    ev = fit_flyby(t, f)
    assert abs(ev.f0 - 90.0) < 0.3
    assert abs(ev.speed - 60.0) < 3.0
    assert abs(ev.dist - 1500.0) < 150.0
    assert abs(ev.t_heard - 61.0) < 0.5


def test_synthesised_tone_follows_model_curve():
    tr = straight_track()
    obs = np.array([0.0, 0.0, 1.5])
    t0 = 900.0
    audio = synthesize(obs, [Source(tr, 100.0, level=0.3, rpm_drift=0.0)], t0, 240.0, hum=0.0)
    spec = compute_spectrogram(audio, 8000)
    times, freqs, _ = trace_ridge(spec, (60, 180), (75, 135))  # 70 m/s sweeps 126 -> 83 Hz
    model = 100.0 * doppler_factor(tr, obs, t0 + times, C)
    assert np.median(np.abs(freqs - model)) < 0.5


def test_implausible_passes_are_named():
    t = np.linspace(-60.0, 60.0, 241)
    f = straight_flyby(t, 100.0, 60.0, 2000.0, 0.0)
    good = Event(t_heard=0.0, f0=100.0, speed=60.0, dist=2000.0, line=(t, f))
    assert implausible(good) is None
    assert "beyond" in implausible(Event(t_heard=0.0, f0=100.0, speed=60.0, dist=12000.0))
    assert "speed" in implausible(Event(t_heard=0.0, f0=100.0, speed=250.0, dist=2000.0))
    stepping = np.where(t < 0, 110.0, 90.0)  # a tone switching between two pitches, not a pass
    assert "traced line" in implausible(Event(t_heard=0.0, f0=100.0, speed=60.0, dist=2000.0, line=(t, stepping)))
