# app.py
import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ---------------------------
# Config / Model file paths
# ---------------------------
STRESS_MODEL_PATH = "/home/marcius/mb_files/19.python/sss/stress_model/if_model.pkl"
STRESS_SCALER_PATH = "/home/marcius/mb_files/19.python/sss/stress_model/if_scaler.pkl"
STALL_MODEL_PATH = "/home/marcius/mb_files/19.python/sss/stall_model/stall_if_model.pkl"
STALL_SCALER_PATH = "/home/marcius/mb_files/19.python/sss/stall_model/stall_if_scaler.pkl"

# stall threshold as in your Block 2
STALL_THRESHOLD = 8

# ---------------------------
# Helper: severity color mapping (dark-theme friendly, white text)
# ---------------------------
SEVERITY_COLORS = {
    0: "#1b5e20",  # deep green
    1: "#f4d35e",  # warm yellow (visible on dark bg)
    2: "#f08a24",  # orange
    3: "#d62828",  # red
    4: "#8b0000",  # dark red
    5: "#6a0dad",  # purple (severe)
}

def severity_color(sev):
    sev = int(np.clip(int(sev), 0, 5))
    return SEVERITY_COLORS.get(sev, "#333333")

# ---------------------------
# Load models (safe fail with informative message)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model/scaler at {path}: {e}")
        return None

stress_model = load_model(STRESS_MODEL_PATH)
stress_scaler = load_model(STRESS_SCALER_PATH)
stall_model = load_model(STALL_MODEL_PATH)
stall_scaler = load_model(STALL_SCALER_PATH)

# ---------------------------
# Dummy data (from Block 3)
# ---------------------------
n1 = 50
n2 = 50

# Base levels
base_value_steam_1 = 120
base_value_steam_2 = 120

base_value_condensate_1 = 100
base_value_condensate_2 = 30

# --- Periods ---
# Period 1: low noise, base level 1
steam_p1 = base_value_steam_1 + np.random.normal(0, 1, n1)
cond_p1 = base_value_condensate_1 + np.random.normal(0, 3, n1)

# Period 2: high noise, base level 1
steam_p2 = base_value_steam_1 + np.random.normal(0, 7, n2)
cond_p2 = base_value_condensate_1 + np.random.normal(0, 10, n2)

# Period 3: low noise, base level 2
steam_p3 = base_value_steam_2 + np.random.normal(0, 1, n1)
cond_p3 = base_value_condensate_2 + np.random.normal(0, 5, n1)

# Period 4: high noise, base level 2
steam_p4 = base_value_steam_2 + np.random.normal(0, 7, n2)
cond_p4 = base_value_condensate_2 + np.random.normal(0, 10, n2)

# --- Combine all periods ---
steam_values = np.concatenate([steam_p1, steam_p2, steam_p3, steam_p4])
condensate_values = np.concatenate([cond_p1, cond_p2, cond_p3, cond_p4])

timestamps = pd.date_range(start='2025-10-10 00:00:00', periods=len(steam_values), freq='min')

df = pd.DataFrame({
    'timestamp': timestamps,
    'steam': steam_values,
    'condensate': condensate_values
})

# engineered features
df['steam_diff'] = df['steam'].diff()
df['condensate_diff'] = df['condensate'].diff()
df['deltaT'] = df['steam'] - df['condensate']
df = df.dropna().reset_index(drop=True)

# storage for scores
df['stall_severity'] = np.nan
df['stall_score'] = np.nan
df['stress_severity'] = np.nan
df['stress_score'] = np.nan

# ---------------------------
# Scoring functions (use the same logic as your Blocks)
# ---------------------------
def compute_stress_severity(steam_diff, condensate_diff):
    if stress_model is None or stress_scaler is None:
        return None, None
    features = np.array([[steam_diff, condensate_diff]])
    score = -stress_model.decision_function(features)[0]
    severity = int(np.round(stress_scaler.transform([[score]])[0][0]))
    severity = int(np.clip(severity, 0, 5))
    return severity, score

def compute_stall_severity(deltaT):
    if stall_model is None or stall_scaler is None:
        return None, None
    features = np.array([[float(deltaT)]])
    score = -stall_model.decision_function(features)[0]
    severity = int(np.round(stall_scaler.transform([[score]])[0][0]))
    # apply stall threshold rule (if small deltaT -> severity 0)
    if abs(deltaT) < STALL_THRESHOLD:
        severity = 0
    severity = int(np.clip(severity, 0, 5))
    return severity, score

# ---------------------------
# Streamlit UI layout
# ---------------------------
st.set_page_config(page_title="Steam/Condensate Simulation", layout="wide", initial_sidebar_state="auto")
st.title("Steam / Condensate - Stall & Stress Simulation")

# Sidebar: controls
with st.sidebar:
    st.markdown("### Simulation controls")
    speed_ms = st.slider("Step delay (ms)", min_value=50, max_value=2000, value=200, step=50)
    autoplay = st.checkbox("Auto-run (loop over dataset until end)", value=False)
    st.markdown("---")
    st.markdown("Model files used:")
    st.code(STRESS_MODEL_PATH + "\n" + STRESS_SCALER_PATH, language="text")
    st.code(STALL_MODEL_PATH + "\n" + STALL_SCALER_PATH, language="text")

start_button = st.button("Start simulation")

# placeholders for dynamic content
label_box = st.empty()
graph_box = st.empty()
numbers_box = st.empty()
progress_box = st.empty()

# Prepare base plot (full series)
def build_figure(current_index=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['steam'],
        mode='lines+markers',
        name='steam',
        line=dict(color='red'),
        marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['condensate'],
        mode='lines+markers',
        name='condensate',
        line=dict(color='purple'),
        marker=dict(size=6)
    ))
    # highlight current point
    if current_index is not None:
        fig.add_trace(go.Scatter(
            x=[df.loc[current_index, 'timestamp']],
            y=[df.loc[current_index, 'steam']],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[df.loc[current_index, 'timestamp']],
            y=[df.loc[current_index, 'condensate']],
            mode='markers',
            marker=dict(size=12, color='purple', symbol='diamond'),
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="timestamp")
    fig.update_yaxes(title_text="Temperature")
    return fig

# Helper to render colored labels above graph (each rectangle 25% width)
def render_labels(stall_sev, stress_sev):
    # two rectangles side-by-side (each 25% width), centered
    stall_color = severity_color(stall_sev) if stall_sev is not None else "#555555"
    stress_color = severity_color(stress_sev) if stress_sev is not None else "#555555"

    html = f"""
    <div style="display:flex; justify-content:center; gap:20px; align-items:center; margin-bottom:10px;">
      <div style="width:25%; min-width:160px; padding:12px; border-radius:8px; background:{stall_color}; color:white; text-align:center; font-weight:600;">
        Stall severity: {stall_sev if stall_sev is not None else 'N/A'}
      </div>
      <div style="width:25%; min-width:160px; padding:12px; border-radius:8px; background:{stress_color}; color:white; text-align:center; font-weight:600;">
        Stress severity: {stress_sev if stress_sev is not None else 'N/A'}
      </div>
    </div>
    """
    label_box.markdown(html, unsafe_allow_html=True)

# Simulation loop (row-by-row)
if start_button:
    # iterate rows
    total = len(df)
    for i in range(total):
        row = df.loc[i]
        # compute severities
        stress_sev, stress_score = compute_stress_severity(row['steam_diff'], row['condensate_diff'])
        stall_sev, stall_score = compute_stall_severity(row['deltaT'])

        # save into df for reference
        df.at[i, 'stress_severity'] = stress_sev
        df.at[i, 'stress_score'] = stress_score
        df.at[i, 'stall_severity'] = stall_sev
        df.at[i, 'stall_score'] = stall_score

        # display numbers: ONLY steam and condensate
        with numbers_box.container():
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.metric(label="Steam (°C)", value=f"{row['steam']:.2f}")
            with c2:
                st.metric(label="Condensate (°C)", value=f"{row['condensate']:.2f}")
            with c3:
                # small legend and index info
                st.markdown(f"**Index:** {i+1} / {total} &nbsp;&nbsp;&nbsp; **Timestamp:** {row['timestamp']}")
                st.markdown(f"**deltaT:** {row['deltaT']:.2f} &nbsp;&nbsp; **steam_diff:** {row['steam_diff']:.2f} &nbsp;&nbsp; **cond_diff:** {row['condensate_diff']:.2f}")

        # render labels (rectangles above graph)
        render_labels(stall_sev, stress_sev)

        # update graph
        fig = build_figure(current_index=i)
        graph_box.plotly_chart(fig, use_container_width=True)

        # progress
        progress_box.progress((i+1)/total)

        # pause
        time.sleep(speed_ms/1000.0)

    # End of run summary
    st.success("Simulation finished.")
    # show a compact DataFrame of results (optional)
    st.markdown("### Results (sample)")
    st.dataframe(df[['timestamp','steam','condensate','stall_severity','stress_severity']].reset_index(drop=True))
else:
    # Show initial state (no simulation run yet)
    render_labels(None, None)
    fig0 = build_figure(current_index=None)
    graph_box.plotly_chart(fig0, use_container_width=True)
    st.info("Press **Start simulation** to run the step-by-step scoring. Use the sidebar to control speed.")

