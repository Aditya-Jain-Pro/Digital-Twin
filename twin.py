"""HridAI core: a synthetic post-discharge heart-failure cohort and the Kalman-filter patient twin.

Synthetic data only. Not a medical device. Decision support for clinicians.
"""
import numpy as np
import pandas as pd

DAYS = 30
HOSP_THRESHOLD = 3.0      # litres of excess fluid at which the synthetic patient is readmitted
ALERT_LEVEL = 2.0         # congestion level the twin warns about
HORIZON = 5               # days ahead the twin looks
CALIB_DAYS = 3            # days at home used to sync the twin to the person
RISK_THRESHOLD = 0.6      # alert when P(fluid >= ALERT_LEVEL within HORIZON days) reaches this; chosen on the dev cohort (seed 7)
DEV_SEED, TEST_SEED = 7, 11
SENSORS = ["weight", "night_hr", "night_hrv", "night_rr", "sleep_eff", "log_steps", "spo2"]
WATCH = SENSORS[1:]
SENSOR_LABELS = {"weight": "Weight (kg)", "night_hr": "Night heart rate (bpm)", "night_hrv": "Night HRV, RMSSD (ms)",
                 "night_rr": "Night breathing rate (/min)", "sleep_eff": "Sleep efficiency (%)",
                 "log_steps": "Steps (log)", "spo2": "Night SpO2 (%)"}
NOISE_SD = {"weight": 0.7, "night_hr": 3.0, "night_hrv": 6.0, "night_rr": 1.0, "sleep_eff": 4.0,
            "log_steps": 0.25, "spo2": 0.8}
# population response of each sensor to 1 litre of excess fluid (twin's prior)
POP_SENS = {"weight": 1.0, "night_hr": 2.7, "night_hrv": -4.0, "night_rr": 1.2, "sleep_eff": -3.0,
            "log_steps": -0.15, "spo2": -0.6}


# ---------------------------------------------------------------- synthetic EHR + wearable cohort
def make_patient(pid, rng):
    ef = int(rng.choice([20, 25, 30, 35, 40, 45, 50, 55, 60]))
    ehr = {
        "pid": pid,
        "age": int(rng.integers(50, 88)),
        "sex": str(rng.choice(["M", "F"], p=[0.6, 0.4])),
        "ef": ef,
        "egfr": int(rng.integers(25, 95)),
        "ntprobnp": int(np.exp(rng.normal(7.6 if ef < 40 else 7.0, 0.6))),   # pg/mL at discharge
        "sodium": int(rng.normal(137, 3)),
        "diabetes": bool(rng.random() < 0.4),
        "hypertension": bool(rng.random() < 0.6),
        "furosemide_mg": int(rng.choice([20, 40, 80])),
        "prior_admissions": int(rng.integers(0, 4)),
        "hf_prs": int(rng.integers(1, 100)),      # synthetic polygenic risk percentile (genetic marker)
    }
    hfref = ef < 40
    # each person's sensors respond differently to fluid; EF shifts heart rate response
    sens = {s: POP_SENS[s] * rng.lognormal(0, 0.35) for s in WATCH}
    sens["weight"] = 1.0
    sens["night_hr"] = rng.normal(3.5 if hfref else 2.0, 0.6)
    base = {"weight": rng.normal(66, 11), "night_hr": rng.normal(68, 7), "night_hrv": rng.normal(30, 6),
            "night_rr": rng.normal(16, 1.5), "sleep_eff": rng.normal(85, 4),
            "log_steps": np.log(rng.uniform(1200, 5500)), "spo2": rng.normal(96, 1)}

    # decompensation risk rises with low EF, poor kidneys, high NT-proBNP, prior admissions, low sodium
    risk = (0.08 + 0.10 * hfref + 0.08 * (ehr["egfr"] < 45) + 0.08 * (ehr["ntprobnp"] > 3000)
            + 0.05 * ehr["prior_admissions"] + 0.05 * (ehr["sodium"] < 134) + 0.03 * (ehr["hf_prs"] > 80))
    hf_event = rng.random() < min(risk, 0.7)
    other_event = (not hf_event) and rng.random() < 0.06     # pneumonia, fall: no fluid signal
    onset = int(rng.integers(5, 25)) if hf_event else None
    drift = rng.uniform(0.15, 0.7)                             # fast decompensators give less warning
    other_day = int(rng.integers(5, DAYS)) if other_event else None

    c, rows, hosp_day, cause = max(rng.normal(0.3, 0.2), 0), [], None, None
    scale_offset = 0.0
    for d in range(DAYS):
        if hf_event and d >= onset:
            c += drift + rng.normal(0, 0.06)
        else:
            c = max(c * 0.8 + rng.normal(0, 0.08), 0)
            if rng.random() < 0.06:          # salty wedding dinner: transient fluid bump that self-corrects
                c += rng.uniform(0.6, 1.3)
        if rng.random() < 0.05:              # weighed on a different scale / with clothes on
            scale_offset = rng.normal(0, 0.8)
        row = {"pid": pid, "day": d, "true_congestion": c}
        for s in SENSORS:
            row[s] = base[s] + sens[s] * c + rng.normal(0, NOISE_SD[s])
        row["weight"] += scale_offset
        if rng.random() < 0.25:
            row["weight"] = np.nan          # elderly patients skip the scale often
        for s in WATCH:
            if rng.random() < 0.10:
                row[s] = np.nan             # watch not worn / not charged
        rows.append(row)
        if c >= HOSP_THRESHOLD:
            hosp_day, cause = d, "heart_failure"
            break
        if other_event and d == other_day:
            hosp_day, cause = d, "other"
            break
    ehr["hosp_day"], ehr["hosp_cause"] = hosp_day, cause
    return ehr, pd.DataFrame(rows)


def make_cohort(n=200, seed=7):
    rng = np.random.default_rng(seed)
    ehrs, series = zip(*(make_patient(i, rng) for i in range(n)))
    return pd.DataFrame(ehrs), pd.concat(series, ignore_index=True)


# ---------------------------------------------------------------- the twin
def ehr_risk(ehr):
    """Discharge-record risk score in [0, 1]; makes the twin quicker to believe a rising trend."""
    z = (-2.2 + 0.9 * (ehr["ef"] < 40) + 0.7 * (ehr["egfr"] < 45) + 0.7 * (ehr["ntprobnp"] > 3000)
         + 0.4 * ehr["prior_admissions"] + 0.4 * (ehr["sodium"] < 134) + 0.3 * (ehr["hf_prs"] > 80))
    return 1 / (1 + np.exp(-z))


def twin_sensitivity(ehr, use_ehr=True):
    """Prior on how each sensor responds to fluid, personalised from the discharge record."""
    s = dict(POP_SENS)
    if use_ehr:
        s["night_hr"] = 3.5 if ehr["ef"] < 40 else 2.0
    return np.array([s[k] for k in SENSORS])


def run_twin(ehr, df, sensors=SENSORS, use_ehr=True, q_level=0.03):
    """Kalman filter over the hidden state [excess fluid (L), daily trend (L/day)].

    Strictly causal: the output for day t uses only data up to day t.
    Returns one row per day: estimate, uncertainty, 5-day risk, fidelity, ask-for-data flag.
    """
    idx = [SENSORS.index(s) for s in sensors]
    H_all = twin_sensitivity(ehr, use_ehr)[idx]
    R_all = np.array([NOISE_SD[s] ** 2 for s in sensors])
    q_trend = 0.01 * (1 + 2 * ehr_risk(ehr)) if use_ehr else 0.015

    calib = df.iloc[:CALIB_DAYS]

    def personal_baseline(i, s):
        vals = calib[s].dropna()
        return vals.mean() - H_all[i] * 0.3 if not vals.empty else np.nan

    baseline = np.array([personal_baseline(i, s) for i, s in enumerate(sensors)])

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level, q_trend])
    Fh = np.linalg.matrix_power(F, HORIZON)
    x, P = np.array([0.3, 0.0]), np.diag([0.1, 0.01])
    out, inside = [], []
    for _, r in df.iterrows():
        x, P = F @ x, F @ P @ F.T + Q
        z = np.array([r[s] for s in sensors], dtype=float)
        # a sensor never read during the sync days gets its baseline from its first later reading
        late = np.isnan(baseline) & ~np.isnan(z) & (r["day"] >= CALIB_DAYS)
        baseline[late] = z[late] - H_all[late] * x[0]
        seen = ~np.isnan(z) & ~np.isnan(baseline) & ~late
        if r["day"] >= CALIB_DAYS and seen.any():
            H = np.column_stack([H_all[seen], np.zeros(seen.sum())])
            y = z[seen] - baseline[seen] - H @ x
            S = H @ P @ H.T + np.diag(R_all[seen])
            # fidelity: did reality land inside the twin's own 95% prediction band?
            inside.extend((np.abs(y) <= 1.96 * np.sqrt(np.diag(S))).tolist())
            K = P @ H.T @ np.linalg.inv(S)
            x, P = x + K @ y, (np.eye(2) - K @ H) @ P
        Ph = Fh @ P @ Fh.T + HORIZON * Q
        mu, sd = (Fh @ x)[0], np.sqrt(Ph[0, 0])
        risk = 0.5 * (1 + erf((mu - ALERT_LEVEL) / (sd * np.sqrt(2))))
        out.append({"day": int(r["day"]), "est": x[0], "est_sd": np.sqrt(P[0, 0]), "trend": x[1],
                    "risk5d": risk, "fidelity": np.mean(inside) if inside else np.nan,
                    "ask_weigh_in": bool("weight" in sensors and np.isnan(r["weight"])
                                         and np.sqrt(P[0, 0]) > 0.35 and x[1] > 0.05)})
    return pd.DataFrame(out)


def erf(x):
    # numpy-only erf (Abramowitz-Stegun 7.1.26), accurate to 1e-7
    s = np.sign(x); x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496735) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def what_if(state_est, state_trend, extra_clearance, days=10):
    """Project excess fluid forward if the clinician adds diuretic (litres/day removed). Illustrative."""
    c, tr, path = state_est, state_trend, []
    for _ in range(days):
        c = max(c + tr - extra_clearance, 0)
        tr = tr * 0.9
        path.append(c)
    return path


# ---------------------------------------------------------------- alerts + evaluation
def alerts_from(series, threshold, refractory=5, start=CALIB_DAYS):
    days, last = [], -99
    for d, v in series:
        if d >= start and v >= threshold and d - last > refractory:
            days.append(d); last = d
    return days


def weight_rule(df):
    """Guideline self-care rule: weight gain > 2 kg within 3 days. Returns (day, 1.0) for each trigger."""
    w = df.set_index("day")["weight"]
    out = []
    for d in w.index:
        win = w.loc[max(0, d - 3):d - 1].dropna()
        if not np.isnan(w.loc[d]) and len(win) and w.loc[d] - win.min() > 2.0:
            out.append((d, 1.0))
    return out


def score(ehrs, alert_days, window=14, min_lead=2):
    """Event-level metrics. alert_days: {pid: [days]}.

    A heart-failure readmission counts as caught only if an alert fired 2-14 days before it,
    enough time for a clinician to act. Other-cause readmissions are not counted as catchable.
    """
    rows = []
    for _, e in ehrs.iterrows():
        al = alert_days[e.pid]
        hd = None if pd.isna(e.hosp_day) else int(e.hosp_day)
        hf = e.hosp_cause == "heart_failure"
        true_al = [a for a in al if hf and hd - window <= a <= hd - min_lead]
        late = [a for a in al if hf and hd - min_lead < a < hd]   # too late to act, but not false
        days = (hd + 1) if hd is not None else DAYS
        rows.append({"hf_event": hf, "caught": bool(true_al), "lead": (hd - true_al[0]) if true_al else np.nan,
                     "false_alerts": len(al) - len(true_al) - len(late), "days": days})
    r = pd.DataFrame(rows)
    return {"hf_readmissions": int(r.hf_event.sum()),
            "sensitivity": r[r.hf_event].caught.mean(),
            "median_lead_days": r[r.caught].lead.median(),
            "false_alerts_per_patient_month": r.false_alerts.sum() / (r.days.sum() / 30)}


def twin_risk(ehrs, series, **kw):
    by_pid = dict(tuple(series.groupby("pid")))
    out = {}
    for _, e in ehrs.iterrows():
        tw = run_twin(e, by_pid[e.pid].reset_index(drop=True), **kw)
        out[e.pid] = list(zip(tw.day.values, tw.risk5d.values))
    return out


def evaluate(ehrs, series, threshold=RISK_THRESHOLD):
    by_pid = dict(tuple(series.groupby("pid")))
    tw = twin_risk(ehrs, series)
    res = {"HridAI twin": score(ehrs, {p: alerts_from(tw[p], threshold) for p in tw}),
           "2 kg / 3 day weight rule": score(ehrs, {p: alerts_from(weight_rule(by_pid[p]), 0.5) for p in tw})}
    return pd.DataFrame(res).T


def ablation(ehrs, series, threshold=RISK_THRESHOLD):
    variants = {
        "All sensors + EHR (HridAI)": {},
        "All sensors, no EHR personalisation": {"use_ehr": False},
        "Smartwatch only + EHR": {"sensors": WATCH},
        "Scale only + EHR": {"sensors": ["weight"]},
    }
    rows = {}
    for name, kw in variants.items():
        tw = twin_risk(ehrs, series, **kw)
        rows[name] = score(ehrs, {p: alerts_from(tw[p], threshold) for p in tw})
    return pd.DataFrame(rows).T


def threshold_sweep(ehrs, series, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    tw = twin_risk(ehrs, series)
    return pd.DataFrame([{"threshold": t, **score(ehrs, {p: alerts_from(tw[p], t) for p in tw})}
                         for t in thresholds])


if __name__ == "__main__":
    # threshold was picked on the dev cohort (seed 7); everything below is the untouched test cohort
    ehrs, series = make_cohort(n=500, seed=TEST_SEED)
    print(f"{len(ehrs)} synthetic patients: {(ehrs.hosp_cause == 'heart_failure').sum()} heart-failure "
          f"readmissions, {(ehrs.hosp_cause == 'other').sum()} other-cause readmissions in 30 days\n")
    print(evaluate(ehrs, series).round(2).to_string(), "\n")
    print(ablation(ehrs, series).round(2).to_string())
