import json

import numpy as np

from doppler_loc.project import Project, Settings, candidate_key, fit_key, rank_candidates


def _project(tmp_path, **settings):
    s = dict(name="t", audio="a.wav", date="2026-01-01", lat=0.0, lon=0.0, spacing_m=200.0, **settings)
    (tmp_path / "project.json").write_text(json.dumps(dict(settings=s, passes=[], adsb=None, verdicts={})))
    return Project(tmp_path)


def test_a_new_project_searches_a_500_m_grid():
    assert Settings(name="t", audio="a.wav", date="2026-01-01", lat=0.0, lon=0.0).spacing_m == 500.0


def test_a_project_with_the_old_fast_switch_still_opens_at_its_coarser_grid(tmp_path):
    assert _project(tmp_path, fast=False).settings.spacing_m == 200.0
    assert _project(tmp_path, fast=True).settings.spacing_m == 1000.0


def _cand(lat, *shares):
    explains = [dict(pass_id=f"P{i}", flight=f"F{i}", share=s) for i, s in enumerate(shares)]
    return dict(pair="P0-P1", lat=lat, lon=0.0, score=0.0, support=round(sum(shares), 2), explains=explains)


def _ranked(cands, **state):
    return rank_candidates(dict(candidates=cands), state)["candidates"]


def test_without_verdicts_candidates_rank_by_the_share_each_flight_follows():
    a, b = _ranked([_cand(1.0, 0.7, 0.2), _cand(2.0, 0.9, 0.8)])
    assert (a["lat"], a["combined"], a["explained"]) == (2.0, 1.7, 2)
    assert (b["lat"], b["combined"], b["explained"]) == (1.0, 0.9, 1)
    assert [x["status"] for x in b["explains"]] == ["accepted", "undecided"]


def test_a_fit_verdict_counts_at_its_own_candidate_only():
    good, other = _cand(2.0, 0.9, 0.8), _cand(1.0, 0.9, 0.7)
    no = {candidate_key(good): {fit_key("P0", "F0"): "rejected"}}
    first, second = _ranked([good, other], fit_verdicts=no)
    assert (first["lat"], first["combined"], first["explained"]) == (1.0, 1.6, 2)  # untouched, and now on top
    assert (second["lat"], second["combined"], second["explained"]) == (2.0, 0.8, 1)
    assert second["explains"][0]["verdict"] == "rejected"
    assert first["explains"][0]["verdict"] is None


def test_yes_counts_in_full_and_unsure_counts_half():
    c = _cand(1.0, 0.4, 0.8)
    verdicts = {candidate_key(c): {fit_key("P0", "F0"): "accepted", fit_key("P1", "F1"): "undecided"}}
    [r] = _ranked([c], fit_verdicts=verdicts)
    assert r["combined"] == 1.4 and r["explained"] == 1


def test_a_verdict_lapses_when_another_flight_takes_the_pass():
    c = _cand(1.0, 0.9)
    [r] = _ranked([c], fit_verdicts={candidate_key(c): {fit_key("P0", "other"): "rejected"}})
    assert r["explains"][0]["verdict"] is None and r["combined"] == 0.9


def _passes(*ids):
    return [dict(id=i, status="accepted") for i in ids]


def test_a_clear_lead_explaining_every_accepted_pass_is_trusted():
    a, b = _ranked([_cand(1.0, 0.95, 0.95, 0.95), _cand(2.0, 0.7, 0.2, 0.0)], passes=_passes("P0", "P1", "P2"))
    assert a["lat"] == 1.0 and a["confidence"] > 0.9
    assert b["confidence"] < 0.05


def test_two_spots_with_the_same_evidence_cannot_both_be_sure():
    a, b = _ranked([_cand(1.0, 0.9, 0.9), _cand(2.0, 0.9, 0.9)], passes=_passes("P0", "P1"))
    assert a["confidence"] + b["confidence"] <= 1.0


def test_a_near_neighbour_is_the_same_answer_and_takes_no_lead_from_it():
    a, near, far = _ranked([_cand(1.0, 0.95, 0.95), _cand(1.001, 0.95, 0.9), _cand(2.0, 0.3, 0.0)], passes=_passes("P0", "P1"))
    assert a["confidence"] > 0.9 and near["confidence"] > 0.9 and far["confidence"] < 0.1


def test_verdicts_move_the_confidence_at_once():
    c, other = _cand(1.0, 0.9, 0.7), _cand(2.0, 0.8, 0.6)
    state = dict(passes=_passes("P0", "P1"))
    before = _ranked([dict(c), dict(other)], **state)[0]["confidence"]
    yes = {candidate_key(c): {fit_key("P0", "F0"): "accepted", fit_key("P1", "F1"): "accepted"}}
    assert _ranked([dict(c), dict(other)], fit_verdicts=yes, **state)[0]["confidence"] > before
    no = {candidate_key(c): {fit_key("P0", "F0"): "rejected"}}
    now = next(x for x in _ranked([dict(c), dict(other)], fit_verdicts=no, **state) if x["lat"] == 1.0)
    assert now["confidence"] < before
    # Saying no to the rival's curves helps too.
    no = {candidate_key(other): {fit_key("P0", "F0"): "rejected"}}
    assert _ranked([dict(c), dict(other)], fit_verdicts=no, **state)[0]["confidence"] > before


def test_a_pass_added_after_the_last_check_waits_for_the_re_check():
    a, *_ = _ranked([_cand(1.0, 0.9, 0.9)], passes=_passes("P0", "P1", "M1"))
    b, *_ = _ranked([_cand(1.0, 0.9, 0.9)], passes=_passes("P0", "P1"))
    assert a["confidence"] == b["confidence"]


def _heard(flight, grade="visible", z=4.0):
    return dict(flight=flight, grade=grade, z=z)


def test_a_flight_heard_and_traced_counts_once():
    c = dict(_cand(1.0, 0.8), heard=[_heard("F0"), _heard("other")])
    [r] = _ranked([c])
    assert r["fit_counted"] == 0.8
    assert r["heard_counted"] == 0.5  # only the other flight, at half
    assert r["combined"] == 1.3


def test_tracing_a_heard_flight_adds_evidence():
    heard_only = dict(_cand(1.0, 0.8), heard=[_heard("new")])
    traced = dict(_cand(1.0, 0.8, 0.9), heard=[_heard("F1")])
    assert _ranked([traced])[0]["combined"] > _ranked([heard_only])[0]["combined"]


def test_a_partial_match_does_not_stop_a_flight_counting_as_heard():
    c = dict(_cand(1.0, 0.3), heard=[_heard("F0")])
    assert _ranked([c])[0]["combined"] == 0.8


def test_saying_a_curve_does_not_fit_also_stops_that_flight_counting_as_heard():
    c = dict(_cand(1.0, 0.8), heard=[_heard("F0")])
    no = {candidate_key(c): {fit_key("P0", "F0"): "rejected"}}
    assert _ranked([c], fit_verdicts=no)[0]["combined"] == 0.0


def test_saying_a_heard_flight_shows_counts_it_in_full():
    c = dict(_cand(1.0, 0.8), heard=[_heard("other")])
    yes = {candidate_key(c): {"other": "accepted"}}
    assert _ranked([c], heard_verdicts=yes)[0]["combined"] == 1.8


def _rec(ident, source, t0=100.0, f0=100.0):
    t = np.arange(t0, t0 + 60.0, 0.5)
    f = f0 - 10.0 * np.tanh((t - t0 - 30.0) / 10.0)
    return dict(id=ident, source=source, status="accepted", kind="likely aircraft", assessed="likely aircraft", confidence=0.9, notes=[], f0=f0,
                line_t=t.tolist(), line_f=f.tolist())


def test_a_detected_pass_sharing_a_retraced_passs_name_is_renamed_on_opening(tmp_path):
    s = dict(name="t", audio="a.wav", date="2026-01-01", lat=0.0, lon=0.0)
    passes = [_rec("P1", "detected"), _rec("P2", "manual", t0=500.0), _rec("P2", "detected", t0=900.0), _rec("M1", "manual", t0=1300.0)]
    (tmp_path / "project.json").write_text(json.dumps(dict(settings=s, passes=passes, adsb=None, verdicts={})))
    p = Project(tmp_path)
    ids = [q["id"] for q in p.state["passes"]]
    assert len(set(ids)) == len(ids)
    assert next(q for q in p.state["passes"] if q["source"] == "manual" and q["line_t"][0] == 500.0)["id"] == "P2"
