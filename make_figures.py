"""Reproduce every number and chart in the README and presentation: python make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twin import make_cohort, evaluate, ablation, threshold_sweep, run_twin, alerts_from, weight_rule, \
    RISK_THRESHOLD, ALERT_LEVEL, TEST_SEED

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)
TWIN, RULE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 200})

ehrs, series = make_cohort(n=500, seed=TEST_SEED)
res, abl, sweep = evaluate(ehrs, series), ablation(ehrs, series), threshold_sweep(ehrs, series)
rule = res.loc["2 kg / 3 day weight rule"]
twin = res.loc["HridAI twin"]

# 1. trade-off curve: twin at every alert threshold vs the guideline rule
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep.false_alerts_per_patient_month, sweep.sensitivity * 100, color=TWIN, lw=2, marker="o", ms=7, label="HridAI twin (each dot = one alert threshold)")
ax.scatter([rule.false_alerts_per_patient_month], [rule.sensitivity * 100], color=RULE, s=110, zorder=5, label="2 kg / 3 day weight rule")
ax.annotate(f"Twin at default threshold\n{twin.sensitivity:.0%} caught, {twin.false_alerts_per_patient_month:.2f} false alerts",
            (twin.false_alerts_per_patient_month, twin.sensitivity * 100), xytext=(1.2, 70), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate(f"Weight rule\n{rule.sensitivity:.0%} caught, {rule.false_alerts_per_patient_month:.2f} false alerts",
            (rule.false_alerts_per_patient_month, rule.sensitivity * 100), xytext=(1.6, 45), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.set_xlabel("False alerts per patient-month (lower is better)")
ax.set_ylabel("Heart-failure readmissions caught 2-14 days ahead (%)")
ax.set_ylim(0, 105); ax.set_xlim(0, max(sweep.false_alerts_per_patient_month) * 1.05)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "tradeoff_curve.png"); plt.close(fig)

# 2. ablation: what each data stream contributes, with today's rule for reference
fig, ax = plt.subplots(figsize=(10, 4.2))
rows = list(abl.iterrows()) + [("2 kg / 3 day weight rule (today)", rule)]
rows = rows[::-1]
for i, (name, r) in enumerate(rows):
    v = r.sensitivity * 100
    ax.barh(i, v, color=RULE if "rule" in name else TWIN, height=0.5)
    ax.text(v + 1.5, i, f"{v:.0f}%  ·  {r.false_alerts_per_patient_month:.2f} false alerts/pt-month", va="center", color=INK, fontsize=11)
ax.set_yticks(range(len(rows)), [n for n, _ in rows])
ax.set_xlim(0, 160); ax.set_xticks(range(0, 101, 20))
ax.set_xlabel("Heart-failure readmissions caught 2-14 days ahead (%)")
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT / "ablation.png"); plt.close(fig)

# 3. one patient's story: twin belief vs the weight rule
caught = ehrs[(ehrs.hosp_cause == "heart_failure")]
pick = None
for _, e in caught.iterrows():
    df = series[series.pid == e.pid].reset_index(drop=True)
    tw = run_twin(e, df)
    ta = alerts_from(zip(tw.day, tw.risk5d), RISK_THRESHOLD)
    ra = alerts_from(weight_rule(df), 0.5)
    if ta and e.hosp_day - ta[0] >= 5 and not [a for a in ra if e.hosp_day - 14 <= a <= e.hosp_day - 2] and len(df) >= 20:
        pick = (e, df, tw, ta, ra); break
e, df, tw, ta, ra = pick
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
a1.fill_between(tw.day, tw.est - 2 * tw.est_sd, tw.est + 2 * tw.est_sd, color=TWIN, alpha=0.18, lw=0)
a1.plot(tw.day, tw.est, color=TWIN, lw=2, label="Twin's estimate of hidden fluid (± 2 sd)")
a1.plot(df.day, df.true_congestion, color=MUTED, lw=1.5, ls=":", label="Hidden truth (synthetic, not visible to the twin)")
a1.axhline(ALERT_LEVEL, color=MUTED, ls="--", lw=1)
a1.set_ylabel("Excess fluid (L)"); a1.legend(frameon=False, loc="upper left", fontsize=10)
a2.plot(df.day, df.weight, color=RULE, marker="o", ms=5, lw=1.5, label="Daily weight (gaps = skipped)")
a2.set_ylabel("Weight (kg)"); a2.set_xlabel("Days after discharge"); a2.legend(frameon=False, loc="upper left", fontsize=10)
for ax in (a1, a2):
    ax.axvline(ta[0], color=TWIN, lw=2); ax.axvline(e.hosp_day, color=INK, lw=2)
a1.text(ta[0] - 0.3, a1.get_ylim()[1] * 0.92, "twin alert", color=TWIN, ha="right", fontsize=11)
a1.text(e.hosp_day + 0.3, a1.get_ylim()[1] * 0.92, "readmitted", color=INK, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "patient_story.png"); plt.close(fig)

summary = {"cohort": {"patients": len(ehrs), "hf_readmissions": int((ehrs.hosp_cause == "heart_failure").sum()),
                      "other_readmissions": int((ehrs.hosp_cause == "other").sum())},
           "threshold": RISK_THRESHOLD, "results": res.round(3).to_dict(orient="index"),
           "ablation": abl.round(3).to_dict(orient="index"), "story_patient": {"pid": int(e.pid),
           "alert_day": int(ta[0]), "readmit_day": int(e.hosp_day), "rule_alerts": [int(a) for a in ra]}}
Path("docs/results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
