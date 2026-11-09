import numpy as np

from doppler_loc import scenarios
from doppler_loc.search import Passes, observer_grid
from doppler_loc.spectro import compute_spectrogram
from doppler_loc.stack import refine_place, spectrogram_peaks, stack_search


def test_stacking_finds_the_place_without_any_pass_traced():
    sc = scenarios.two_passes()
    peaks = spectrogram_peaks(compute_spectrogram(sc.audio, sc.sr))
    grid = observer_grid((0.0, 0.0), 10000.0, 1000.0, sc.ground)
    sm = stack_search(sc.adsb, grid, peaks, (sc.t_rec + 12.0, 30.0), passes=Passes(sc.adsb, grid), workers=2)
    k = sm.places(grid)[0]
    assert np.hypot(*(grid[k, :2] - sc.truth[:2])) < 800.0  # the 1 km grid point nearest the truth wins

    def place(x, y):
        return np.stack([x, y, sc.ground(x, y) + 2.0], -1)

    obs, start, score, heard = refine_place(sc.adsb, grid[k, :2], sm.start[k], peaks, sm.stats, place=place)
    assert np.hypot(*(obs[:2] - sc.truth[:2])) < 150.0
    assert abs(start - sc.t_rec) < 3.0
    # Both planes heard (and only they), with their emitted pitches, though none was traced.
    assert sorted(sc.adsb[i].flight_id for i, *_ in heard) == ["N2209K", "N738QA"]
    pitch = {sc.adsb[i].flight_id: f for i, _, f, _ in heard}
    assert abs(pitch["N738QA"] / sc.f0s[0] - 1) < 0.02 and abs(pitch["N2209K"] / sc.f0s[1] - 1) < 0.02
