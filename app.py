"""HridAI clinician dashboard: the 30-day heart-failure discharge twin. Run: streamlit run app.py"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from twin import (make_cohort, run_twin, weight_rule, alerts_from, what_if, evaluate, ehr_risk,
                  ALERT_LEVEL, HOSP_THRESHOLD, RISK_THRESHOLD, HORIZON, SENSOR_LABELS, DEV_SEED)

st.set_page_config(page_title="HridAI twin", page_icon="🫀", layout="wide")


@st.cache_data
def cohort():
    return make_cohort(n=200, seed=DEV_SEED)


@st.cache_data
def cohort_results():
    return evaluate(*make_cohort(n=200, seed=DEV_SEED)).round(2)


ehrs, series = cohort()
st.title("HridAI · heart-failure discharge twin")
st.caption("A virtual replica of each patient for the 30 days after a heart-failure discharge. "
           "Synthetic patients only. Decision support for clinicians, not a medical device.")

with st.sidebar:
    view = st.radio("Patients", ["Heart-failure readmissions", "Everyone"], horizontal=False)
    pool = ehrs[ehrs.hosp_cause == "heart_failure"] if view != "Everyone" else ehrs
    pid = st.selectbox("Patient ID", pool.pid.tolist())
    e = ehrs.set_index("pid").loc[pid].to_dict() | {"pid": pid}
    df = series[series.pid == pid].reset_index(drop=True)
    day = st.slider("Replay to day after discharge", 0, len(df) - 1, len(df) - 1)
    reveal = st.checkbox("Reveal hidden ground truth (demo only)", False)
    st.markdown("**Discharge record (EHR)**")
    st.table({"Age / sex": f"{e['age']} / {e['sex']}", "Ejection fraction": f"{e['ef']}%",
              "eGFR": str(e["egfr"]), "NT-proBNP": f"{e['ntprobnp']} pg/mL", "Sodium": f"{e['sodium']} mmol/L",
              "Diabetes / HTN": f"{'Y' if e['diabetes'] else 'N'} / {'Y' if e['hypertension'] else 'N'}",
              "Furosemide": f"{e['furosemide_mg']} mg", "Prior HF admissions": str(e["prior_admissions"]),
              "Polygenic risk (pctile)": str(e["hf_prs"]), "EHR risk score": f"{ehr_risk(e):.0%}"})

tw = run_twin(e, df)
now = tw.iloc[day]
tw_alerts = [a for a in alerts_from(zip(tw.day, tw.risk5d), RISK_THRESHOLD) if a <= day]
rule_alerts = [a for a in alerts_from(weight_rule(df), 0.5) if a <= day]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Hidden fluid (estimate)", f"{max(now.est, 0):.1f} L", f"{now.trend:+.2f} L/day", delta_color="inverse")
c2.metric(f"Risk of ≥{ALERT_LEVEL:.0f} L within {HORIZON} days", f"{now.risk5d:.0%}")
c3.metric("Twin fidelity", "syncing" if np.isnan(now.fidelity) else f"{now.fidelity:.0%}",
          help="Share of real readings that fell inside the twin's own 95% prediction band")
c4.metric("First twin alert", f"day {tw_alerts[0]}" if tw_alerts else "none",
          f"2 kg rule: day {rule_alerts[0]}" if rule_alerts else "2 kg rule: none", delta_color="off")

if now.risk5d >= RISK_THRESHOLD:
    st.error(f"Alert: fluid likely to reach {ALERT_LEVEL:.0f} L within {HORIZON} days. Review and call the patient.")
if now.ask_weigh_in:
    st.warning("The twin is unsure and the trend is rising: it asks the family for a weigh-in today.")

seen = df.iloc[: day + 1]
t = tw.iloc[: day + 1]
extra = st.multiselect("Extra signals to show", [k for k in SENSOR_LABELS if k not in ("weight", "night_hr")],
                       default=["night_hrv"], format_func=SENSOR_LABELS.get)
panels = ["weight", "night_hr"] + extra
fig = make_subplots(rows=1 + len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                    subplot_titles=["Twin's belief about hidden fluid, litres (± 2 sd)"] + [SENSOR_LABELS[p] for p in panels])
fig.add_trace(go.Scatter(x=list(t.day) + list(t.day[::-1]), y=list(t.est + 2 * t.est_sd) + list((t.est - 2 * t.est_sd)[::-1]),
                         fill="toself", line=dict(width=0), opacity=0.25, name="uncertainty"), 1, 1)
fig.add_trace(go.Scatter(x=t.day, y=t.est, name="twin estimate"), 1, 1)
if reveal:
    fig.add_trace(go.Scatter(x=seen.day, y=seen.true_congestion, name="hidden truth", line=dict(dash="dot")), 1, 1)
fig.add_hline(y=ALERT_LEVEL, line_dash="dash", row=1, col=1)
for i, p in enumerate(panels, start=2):
    fig.add_trace(go.Scatter(x=seen.day, y=seen[p], mode="markers+lines", name=SENSOR_LABELS[p]), i, 1)
for a in tw_alerts:
    fig.add_vline(x=a, line_color="red", annotation_text="twin alert", annotation_position="top left")
for a in rule_alerts:
    fig.add_vline(x=a, line_color="gray", line_dash="dot", annotation_text="2 kg rule", annotation_position="bottom right")
if not np.isnan(e["hosp_day"]) and day >= e["hosp_day"]:
    fig.add_vline(x=e["hosp_day"], line_color="black", annotation_text=f"readmitted ({e['hosp_cause'].replace('_', ' ')})")
fig.update_layout(height=230 * (1 + len(panels)), margin=dict(t=40, b=10), showlegend=False)
fig.update_xaxes(title_text="days after discharge", row=1 + len(panels), col=1)
st.plotly_chart(fig, width="stretch")

st.subheader("What if the clinician acts today?")
dose = st.select_slider("Extra fluid removed by a diuretic increase (L/day, illustrative)", [0.0, 0.2, 0.4, 0.6], 0.4)
base, act = what_if(now.est, now.trend, 0.0), what_if(now.est, now.trend, dose)
wf = go.Figure([go.Scatter(y=base, name="no change"), go.Scatter(y=act, name=f"remove {dose} L/day")])
wf.add_hline(y=HOSP_THRESHOLD, line_dash="dash", annotation_text="likely readmission")
wf.update_layout(height=280, xaxis_title="days from today", yaxis_title="excess fluid (L)", margin=dict(t=10))
st.plotly_chart(wf, width="stretch")
st.caption("The twin never changes treatment. It shows the clinician a projection; the clinician decides.")

st.subheader("Message to the family caregiver")
if now.risk5d >= RISK_THRESHOLD:
    st.info("**EN:** Papa's body is holding extra water. Please weigh him tomorrow morning and keep salt low. "
            "The doctor's team will call you today.\n\n**HI:** पापा के शरीर में पानी रुक रहा है। कल सुबह उनका वज़न लें "
            "और नमक कम रखें। डॉक्टर की टीम आज आपको फ़ोन करेगी।")
elif now.ask_weigh_in:
    st.info("**EN:** Please weigh Papa tomorrow morning before breakfast.\n\n**HI:** कृपया कल सुबह नाश्ते से पहले पापा का वज़न लें।")
else:
    st.success("**EN:** Recovery is on track. Keep up the morning weigh-ins.\n\n**HI:** रिकवरी ठीक चल रही है। सुबह वज़न लेते रहें।")

with st.expander("Cohort results on these 200 patients: twin vs the 2 kg guideline rule"):
    st.dataframe(cohort_results())
