import numpy as np
import pandas as pd
import pytest

from twin import (make_cohort, run_twin, evaluate, weight_rule, alerts_from, what_if, ehr_risk,
                  SENSORS, CALIB_DAYS)


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(n=40, seed=3)


def patient(cohort, pid):
    ehrs, series = cohort
    return ehrs.set_index("pid").loc[pid].to_dict() | {"pid": pid}, series[series.pid == pid].reset_index(drop=True)


def test_cohort_is_deterministic():
    a, b = make_cohort(n=10, seed=5), make_cohort(n=10, seed=5)
    pd.testing.assert_frame_equal(a[1], b[1])


def test_vitals_in_plausible_ranges(cohort):
    s = cohort[1]
    assert s.night_hr.dropna().between(30, 160).all()
    assert s.spo2.dropna().between(80, 101).all()
    assert s.weight.dropna().between(25, 140).all()


def test_twin_is_causal(cohort):
    """Output on day t must not change when future days are removed (no look-ahead leakage)."""
    ehrs, _ = cohort
    for pid in ehrs.pid[:10]:
        e, df = patient(cohort, pid)
        full = run_twin(e, df)
        for t in range(CALIB_DAYS + 1, len(df), 4):
            part = run_twin(e, df.iloc[: t + 1])
            np.testing.assert_allclose(part.risk5d.values, full.risk5d.values[: t + 1])


def test_twin_survives_missing_data(cohort):
    e, df = patient(cohort, 0)
    df = df.copy()
    df.loc[CALIB_DAYS:, SENSORS[1:]] = np.nan     # watch never worn after sync
    tw = run_twin(e, df)
    assert tw.risk5d.notna().all() and tw.est_sd.iloc[-1] > 0


def test_uncertainty_grows_without_data(cohort):
    e, df = patient(cohort, 1)
    df = df.copy()
    df.loc[CALIB_DAYS + 2:, SENSORS] = np.nan
    tw = run_twin(e, df)
    assert tw.est_sd.iloc[-1] > tw.est_sd.iloc[CALIB_DAYS + 2]


def test_ehr_risk_monotone():
    low = {"ef": 55, "egfr": 80, "ntprobnp": 800, "prior_admissions": 0, "sodium": 139, "hf_prs": 30}
    high = {"ef": 25, "egfr": 35, "ntprobnp": 6000, "prior_admissions": 3, "sodium": 131, "hf_prs": 90}
    assert ehr_risk(high) > ehr_risk(low)


def test_weight_rule_fires_on_2kg_gain():
    df = pd.DataFrame({"day": range(6), "weight": [70, 70.2, 70.1, 72.5, np.nan, 70]})
    assert [d for d, _ in weight_rule(df)] == [3]


def test_alert_refractory_period():
    assert alerts_from([(d, 1.0) for d in range(20)], 0.5, refractory=5, start=3) == [3, 9, 15]


def test_what_if_more_diuretic_means_less_fluid():
    assert what_if(2.0, 0.2, 0.4)[-1] < what_if(2.0, 0.2, 0.0)[-1]


def test_twin_beats_weight_rule(cohort):
    ehrs, series = make_cohort(n=150, seed=21)
    r = evaluate(ehrs, series)
    assert r.loc["HridAI twin", "sensitivity"] > r.loc["2 kg / 3 day weight rule", "sensitivity"]


def test_dashboard_data_builds():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("build_dashboard", pathlib.Path(__file__).parents[1] / "dashboard" / "build_dashboard.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = mod.build(n=12)
    assert len(data["patients"]) == 12
    p = data["patients"][0]
    assert len(p["twin"]["risk"]) == len(p["truth"]) == len(p["obs"]["weight"])
