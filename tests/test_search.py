from dataclasses import replace

import numpy as np

from doppler_loc import scenarios
from doppler_loc.measure import fit_flyby, trace_ridge
from doppler_loc.search import (
    Passes, explain_passes, observer_grid, refine_known_start, refine_single, search_event_pairs, search_known_start,
    search_known_time, search_two_events, vote_passes,
)
from doppler_loc.spectro import compute_spectrogram


def locate(sc, clock_error=2.0):
    spec = compute_spectrogram(sc.audio, sc.sr)
    event = fit_flyby(*trace_ridge(spec, *sc.marks[0]))
    t0 = sc.t_rec + clock_error
    grid = observer_grid((0.0, 0.0), 3000.0, 200.0, sc.ground)
    res = search_known_time(spec, sc.adsb, grid, event, t0, clock_tol=4.0)
    obs, s = refine_single(spec, sc.adsb[0], grid[res.best, :2], event, t0, ground=sc.ground, clock_tol=4.0)
    j = int(np.argmax(s.score))
    return grid, res, obs[j], s.shift[j] * spec.hop


def test_maneuvering_pass_is_located():
    sc = scenarios.single_pass(maneuvering=True, duration=240.0)
    _, _, best, clock = locate(sc)
    assert np.linalg.norm(best[:2] - sc.truth[:2]) < 150.0
    assert abs(clock - (-2.0)) <= 1.0


def test_straight_pass_is_mirror_ambiguous():
    sc = scenarios.single_pass(maneuvering=False, duration=240.0)
    grid, res, _, _ = locate(sc)
    tr = sc.adsb[0]
    # Score of the best candidate on each side of the flight path.
    direction = tr.pos[-1, :2] - tr.pos[0, :2]
    rel = grid[:, :2] - tr.pos[0, :2]
    side = np.sign(direction[0] * rel[:, 1] - direction[1] * rel[:, 0])
    left = np.nanmax(np.where(side > 0, res.score, np.nan))
    right = np.nanmax(np.where(side < 0, res.score, np.nan))
    assert min(left, right) / max(left, right) > 0.9


def test_two_passes_found_without_time():
    sc = scenarios.two_passes(n_decoys=15)  # building decoy ADS-B dominates the run time
    spec = compute_spectrogram(sc.audio, sc.sr)
    e1, e2 = (fit_flyby(*trace_ridge(spec, *mark)) for mark in sc.marks)
    grid = observer_grid((0.0, 0.0), 10000.0, 200.0, sc.ground)
    res = search_two_events(spec, sc.adsb, grid, e1, e2, sc.duration, workers=2)
    r = res.best_per_pair()[0]
    assert (res.a[r], res.b[r]) == (0, 1)  # the two flights that made the sounds
    assert np.linalg.norm(grid[res.point[r], :2] - sc.truth[:2]) < 300.0
    assert abs(res.t0[r] - sc.t_rec) < 2.0

    # The all-pairs route, events out of order plus a made-up third one (event 1's line moved 400 s
    # later, when nothing passed the true spot): the real pair should have the most evidence.
    fake = replace(e1, t_heard=e1.t_heard + 400.0, line=(e1.line[0] + 400.0, e1.line[1]))
    found = search_event_pairs(spec, sc.adsb, grid, [e2, e1, fake], sc.duration)
    assert len(found) == 3
    assert (found[0].i, found[0].j) == (1, 0)
    assert found[0].best > found[1].best
    np.testing.assert_allclose(np.sort(found[0].result.total), np.sort(res.total))

    # The true place and start explain both passes with the flights that made them; a place 4 km
    # away at the same start explains less.
    at_truth = explain_passes(sc.adsb, sc.truth, sc.t_rec, [e1, e2])
    assert [x.flight for x in at_truth] == [0, 1]
    assert all(x.share > 0.7 for x in at_truth)
    elsewhere = explain_passes(sc.adsb, sc.truth + np.array([4000.0, 0.0, 0.0]), sc.t_rec, [e1, e2])
    assert sum(x.share for x in elsewhere) < 0.5 * sum(x.share for x in at_truth)

    # With the recording's start known (to 30 s), one pass is enough.
    one = search_known_start(spec, sc.adsb, grid, e1, (sc.t_rec + 12.0, 30.0), passes=Passes(sc.adsb, grid))
    r = one.best_per_flight()[0]
    assert one.flight[r] == 0
    obs, sc1 = refine_known_start(spec, sc.adsb[0], grid[one.point[r], :2], e1, (sc.t_rec + 12.0, 30.0),
                                  ground=sc.ground)
    k = int(np.argmax(sc1.score))
    assert np.linalg.norm(obs[k, :2] - sc.truth[:2]) < 300.0
    assert abs(sc.t_rec + 12.0 + sc1.shift[k] * spec.hop - sc.t_rec) < 3.0

    # Every pass votes, the made-up one too: the true place and start get both real passes' votes.
    passes = Passes(sc.adsb, grid)
    votes = vote_passes(sc.adsb, grid, [fake, e2, e1], (sc.t_rec + 12.0, 30.0), passes=passes)
    assert np.linalg.norm(grid[votes.point[0], :2] - sc.truth[:2]) < 300.0
    assert sorted(votes.passes[0]) == [1, 2] and sorted(votes.flights[0]) == [0, 1]
    assert abs(votes.start[0] - sc.t_rec) < 3.0
    assert votes.votes[0] > 1.4
    # Each real pass is matched with its own flight somewhere; a pass left out of the vote (as a
    # rejected one is) still gets its match, and adds nothing to the count.
    assert [b[1] for b in votes.best[1:]] == [1, 0] and min(b[0] for b in votes.best[1:]) > 0.7
    left_out = vote_passes(sc.adsb, grid, [fake, e2, e1], (sc.t_rec + 12.0, 30.0), passes=passes,
                           voting=[True, True, False])
    assert left_out.best[2][1] == 0 and all(2 not in ps for ps in left_out.passes)
