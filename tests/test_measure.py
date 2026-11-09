import numpy as np

from doppler_loc.measure import harmonics_of, sharpen_with_harmonics, straight_flyby, trace_by_hand, trace_ridge, trim_tails
from doppler_loc.spectro import Spectrogram


def _spec_with_tone():
    """A faint falling line, 100 -> 88 Hz over 120 s, crossed by a much brighter steady tone at 96 Hz,
    with a gap in the line from 60 to 70 s."""
    t = np.arange(0, 120, 0.5)
    f = np.arange(0, 200, 0.5)
    line = 100 - 0.1 * t
    score = np.zeros((len(f), len(t)), np.float32)
    for k, fc in enumerate(line):
        if not 60 <= t[k] < 70:
            score[np.abs(f - fc) < 0.3, k] = 8.0
    score[np.abs(f - 96) < 0.3, :] = 20.0
    return Spectrogram(t, f, score, score), t, line


def test_hand_trace_keeps_to_the_clicks():
    spec, t, line = _spec_with_tone()
    clicks = [(tc, 100 - 0.1 * tc) for tc in range(0, 120, 12)] + [(119.5, 100 - 0.1 * 119.5)]
    times, freqs, strength = trace_by_hand(spec, clicks)
    truth = np.interp(times, t, line)
    # Never onto the tone (it's 12 dB brighter, so it can hold the trace about a hertz off where
    # it crosses the line, no more), and the gap filled from the clicks.
    assert np.abs(freqs - truth).max() <= 1.5
    assert np.abs(freqs - truth).mean() <= 0.3
    gap = (times >= 61) & (times < 69)
    assert np.all(strength[gap] == 3.0)
    # The old guided trace, a +-6 Hz band around straight segments, jumps onto the tone.
    old_t, old_f, _ = trace_ridge(spec, (0, 119.5), (70, 115), guide=clicks)
    assert np.abs(old_f - np.interp(old_t, t, line)).max() > 3.0


def _with_harmonics(ratios, noise=1.0, seed=3):
    """A spectrogram of a falling line and copies of it at `ratios`."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 200.0, 0.5)
    f_axis = np.arange(20.0, 600.0, 0.5)
    pitch = 100.0 - 15.0 * np.tanh((t - 100.0) / 25.0)
    score = np.clip(rng.normal(0.0, noise, (f_axis.size, t.size)), 0.0, None).astype(np.float32)
    for r in (1.0, *ratios):
        rows = np.clip(np.searchsorted(f_axis, r * pitch), 0, f_axis.size - 1)
        score[rows, np.arange(t.size)] = 12.0
    return Spectrogram(t, f_axis, score, score), t, pitch


def test_harmonics_are_found_at_real_ratios_only():
    spec, t, pitch = _with_harmonics((2.0, 3.0))
    found = {x[0] for x in harmonics_of(spec, t, pitch)}
    assert {2.0, 3.0} <= found
    assert 2.5 not in found and 1.5 not in found


def test_a_line_is_sharpened_by_its_harmonics():
    spec, t, pitch = _with_harmonics((2.0, 3.0, 4.0))
    rough = pitch + np.random.default_rng(1).normal(0.0, 0.8, t.size)
    sharper, ratios = sharpen_with_harmonics(spec, t, rough)
    assert {2.0, 3.0, 4.0} <= set(ratios)
    assert np.median(np.abs(sharper - pitch)) < 0.6 * np.median(np.abs(rough - pitch))


def test_one_harmonic_alone_changes_nothing():
    spec, t, pitch = _with_harmonics((2.0,))
    rough = pitch + np.random.default_rng(1).normal(0.0, 0.3, t.size)
    same, ratios = sharpen_with_harmonics(spec, t, rough)
    assert ratios == [2.0] and np.array_equal(same, rough)  # found, to show, but the line is left alone


def _flyby_line():
    t = np.arange(0.0, 240.0, 0.5)
    return t, straight_flyby(t, 100.0, 60.0, 1000.0, 120.0)


def test_a_tail_that_leaves_the_curve_is_cut():
    t, f = _flyby_line()
    bad = f.copy()
    bad[-60:] += np.linspace(4.0, 12.0, 60)  # the last 30 s wander off onto something else
    a, b = trim_tails(t, bad, np.full(t.size, 12.0))
    assert a == 0 and len(t) - 70 <= b <= len(t) - 50


def test_a_faint_tail_on_the_curve_is_kept():
    t, f = _flyby_line()
    strength = np.full(t.size, 15.0)
    strength[:80] = strength[-80:] = 6.0
    assert trim_tails(t, f, strength) == (0, len(t))


def test_a_tail_down_in_the_noise_is_cut_even_on_the_curve():
    t, f = _flyby_line()
    strength = np.full(t.size, 15.0)
    strength[-60:] = 3.0
    a, b = trim_tails(t, f, strength)
    assert a == 0 and len(t) - 70 <= b <= len(t) - 50


def test_a_turning_line_is_left_alone():
    t, _ = _flyby_line()
    f = 100.0 + 8.0 * np.sin(t / 15.0)  # no straight pass follows this
    assert trim_tails(t, f, np.full(t.size, 12.0)) == (0, len(t))
