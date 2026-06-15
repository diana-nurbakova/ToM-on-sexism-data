"""Tests for divergences, F1, Format-C aggregation, and hierarchical consistency."""

import math

from llm_audit import metrics as M


def test_jsd_identical_is_zero():
    assert M.jsd([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_jsd_disjoint_is_one_bit():
    # Maximally separated distributions -> JSD = 1 bit.
    assert math.isclose(M.jsd([1.0, 0.0], [0.0, 1.0]), 1.0, rel_tol=1e-9)


def test_kl_known_value():
    # KL([1,0]||[0.5,0.5]) = 1 bit.
    assert math.isclose(M.kl_divergence([1.0, 0.0], [0.5, 0.5]), 1.0, rel_tol=1e-6)


def test_cross_entropy_known_value():
    # H([1,0],[0.5,0.5]) = 1 bit.
    assert math.isclose(M.cross_entropy([1.0, 0.0], [0.5, 0.5]), 1.0, rel_tol=1e-6)


def test_f1_perfect_and_partial():
    res = M.f1_scores(["sexist", "not_sexist"], ["sexist", "not_sexist"],
                      ["sexist", "not_sexist"])
    assert res["macro_f1"] == 1.0
    res2 = M.f1_scores(["sexist", "sexist"], ["sexist", "not_sexist"],
                       ["sexist", "not_sexist"])
    assert res2["per_class"]["sexist"]["recall"] == 0.5


def test_format_c_single_aggregation():
    soft = M.aggregate_format_c_single(["sexist", "sexist", "not_sexist"], "1.1")
    assert math.isclose(soft["sexist"], 2 / 3)
    assert math.isclose(soft["not_sexist"], 1 / 3)


def test_format_c_multi_aggregation():
    draws = [["objectification"], ["objectification", "misogyny_non_sexual_violence"], []]
    soft = M.aggregate_format_c_multi(draws)
    assert math.isclose(soft["objectification"], 2 / 3)
    assert math.isclose(soft["misogyny_non_sexual_violence"], 1 / 3)
    assert soft["sexual_violence"] == 0.0


def test_hierarchical_consistency_flags_excess():
    res = M.hierarchical_consistency(
        p_not_sexist_11=0.2, p_not_sexist_12=0.5,
        category_mass_13=0.9, p_sexist_11=0.8)
    assert math.isclose(res["not_sexist_gap_11_12"], 0.3)
    # category mass (0.9) exceeds P(sexist)=0.8 by 0.1.
    assert math.isclose(res["category_excess_over_sexist"], 0.1)


def test_assemble_model_soft_from_records():
    # Two Format-C 1.1 draws (parsed_ok) + one refusal -> proportions over 2 valid.
    recs = [
        _rec("C", "1.1", "100", 0, "parsed_ok", "sexist"),
        _rec("C", "1.1", "100", 1, "parsed_ok", "not_sexist"),
        _rec("C", "1.1", "100", 2, "refusal", None),
    ]
    soft, status = M.assemble_model_soft(recs)
    entry = soft[("bare", "1.1", "C", "100")]
    assert entry["n_valid"] == 2
    assert math.isclose(entry["soft"]["sexist"], 0.5)
    assert status[("bare", "1.1", "C")]["refusal"] == 1


def test_k_cap_limits_format_c_draws():
    recs = [_rec("C", "1.1", "100", k, "parsed_ok", "sexist" if k < 2 else "not_sexist")
            for k in range(6)]
    full, _ = M.assemble_model_soft(recs)
    capped, _ = M.assemble_model_soft(recs, k_cap=3)
    assert full[("bare", "1.1", "C", "100")]["n_valid"] == 6
    # first 3 draws = sexist, sexist, not_sexist -> 2/3 sexist
    e = capped[("bare", "1.1", "C", "100")]
    assert e["n_valid"] == 3 and math.isclose(e["soft"]["sexist"], 2 / 3)


def test_cross_lane_compare_matches_format_and_k_on_shared_items(tmp_path):
    import json
    # two lanes, Format C, item "1" shared; lane A also has an unshared item "2".
    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    a = tmp_path / "a.jsonl"; b = tmp_path / "b.jsonl"
    rows_a = [_rec("C", "1.1", it, k, "parsed_ok", "sexist") for it in ("1", "2") for k in range(6)]
    rows_b = [_rec("C", "1.1", "1", k, "parsed_ok", "sexist") for k in range(3)]
    write(a, rows_a); write(b, rows_b)
    gold = {"1": {"1.1": {"sexist": 1.0, "not_sexist": 0.0}, "1.2": {}, "1.3": {}}}
    res = M.cross_lane_compare({"ow": str(a), "prop": str(b)}, gold, k=3, spec="bare")
    assert res["1.1"]["n_shared"] == 1 and res["1.1"]["k"] == 3
    assert set(res["1.1"]["mean_jsd"]) == {"ow", "prop"}  # both lanes scored on shared item


def _rec(fmt, sub, item, k, status, val):
    return {
        "experiment": "E0", "condition": "bare", "subtask": sub, "language": "en",
        "format": fmt, "item_id": item, "k_index": k,
        "parse_status": status, "parsed_value": val,
    }
