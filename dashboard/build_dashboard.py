"""Build the self-contained browser dashboard: python dashboard/build_dashboard.py

Runs the twin on the synthetic demo cohort and embeds the results into dashboard/index.html,
so the page opens in any browser with no install.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from twin import (make_cohort, run_twin, alerts_from, weight_rule, ehr_risk, SENSORS,  # noqa: E402
                  RISK_THRESHOLD, ALERT_LEVEL, HOSP_THRESHOLD, HORIZON, CALIB_DAYS, DEV_SEED)


def r(v, nd=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def build(n=200):
    ehrs, series = make_cohort(n=n, seed=DEV_SEED)
    patients = []
    for _, e in ehrs.iterrows():
        df = series[series.pid == e.pid].reset_index(drop=True)
        tw = run_twin(e, df)
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in e.items() if k not in ("hosp_day", "hosp_cause")}
        p["ehr_risk"] = r(ehr_risk(e), 3)
        p["hosp_day"] = None if np.isnan(e.hosp_day) else int(e.hosp_day)
        p["hosp_cause"] = e.hosp_cause
        p["twin_alerts"] = [int(a) for a in alerts_from(zip(tw.day, tw.risk5d), RISK_THRESHOLD)]
        p["rule_alerts"] = [int(a) for a in alerts_from(weight_rule(df), 0.5)]
        p["truth"] = [r(v) for v in df.true_congestion]
        p["obs"] = {s: [r(v, 3 if s == "log_steps" else 1) for v in df[s]] for s in SENSORS}
        p["twin"] = {"est": [r(v) for v in tw.est], "sd": [r(v, 3) for v in tw.est_sd],
                     "trend": [r(v, 3) for v in tw.trend], "risk": [r(v, 3) for v in tw.risk5d],
                     "fidelity": [r(v, 3) for v in tw.fidelity], "ask": [bool(v) for v in tw.ask_weigh_in]}
        patients.append(p)
    results = json.loads((HERE.parent / "docs" / "results.json").read_text())
    return {"params": {"risk_threshold": RISK_THRESHOLD, "alert_level": ALERT_LEVEL, "hosp_threshold": HOSP_THRESHOLD,
                       "horizon": HORIZON, "calib_days": CALIB_DAYS},
            "results": results, "patients": patients}


if __name__ == "__main__":
    data = json.dumps(build(), separators=(",", ":"))
    body = (HERE / "template.html").read_text().replace("/*__DATA__*/null", data)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    (HERE / "index.html").write_text(page)
    print(f"wrote dashboard/index.html ({len(page) / 1024:.0f} KB)")
