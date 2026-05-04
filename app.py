from pathlib import Path
import time

import pandas as pd
import streamlit as st

from env.smart_home_env import SmartHomeEnv
from smart_rl_runtime import ACTION_NAMES, get_valid_actions, load_agent, run_episode

DEFAULT_MODEL_PATH = Path("artifacts/q_agent.npz")


st.set_page_config(page_title="Smart RL Dashboard", page_icon="⚡", layout="wide")

st.title("Smart RL Dashboard")
st.caption("Run the trained Q-learning agent on synthetic smart-home data and watch the cost change live.")

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Synthetic day seed", min_value=0, max_value=100000, value=42, step=1)
    speed_ms = st.slider("Delay between steps (ms)", min_value=0, max_value=1000, value=200, step=50)
    baseline_choice = st.selectbox("Baseline comparison", ["None", "Idle", "ToU"], index=1)
    model_path_text = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    uploaded_model = st.file_uploader("Or upload a trained model (.npz)", type=["npz"])

    load_clicked = st.button("Load agent")


if "agent" not in st.session_state:
    st.session_state.agent = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if uploaded_model is not None:
    temp_path = Path("artifacts/uploaded_q_agent.npz")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded_model.getvalue())
    try:
        st.session_state.agent = load_agent(temp_path)
        st.session_state.model_loaded = True
        st.success("Uploaded model loaded.")
    except Exception as exc:
        st.session_state.agent = None
        st.session_state.model_loaded = False
        st.error(f"Could not load uploaded model: {exc}")

if load_clicked:
    try:
        candidate_path = Path(model_path_text)
        if candidate_path.exists():
            st.session_state.agent = load_agent(candidate_path)
            st.session_state.model_loaded = True
            st.success(f"Loaded model from {candidate_path}")
        else:
            st.warning(f"Model file not found: {candidate_path}")
    except Exception as exc:
        st.session_state.agent = None
        st.session_state.model_loaded = False
        st.error(f"Could not load model: {exc}")

if st.session_state.agent is None:
    st.info("Load a trained model to begin. If you have not trained yet, run `python train.py` first.")
    st.stop()

agent = st.session_state.agent

col1, col2, col3, col4 = st.columns(4)
col1.metric("Q-table shape", "24×11×5×4×3×5")
col2.metric("Epsilon", f"{agent.epsilon:.3f}")
col3.metric("Alpha", f"{agent.alpha:.2f}")
col4.metric("Gamma", f"{agent.gamma:.2f}")

run_button = st.button("Run live day")

if run_button:
    env = SmartHomeEnv(seed=int(seed))
    state = env.reset()
    baseline_policy = None
    if baseline_choice == "Idle":
        baseline_policy = "idle"
    elif baseline_choice == "ToU":
        baseline_policy = "tou"

    baseline = run_episode(agent, seed=int(seed), policy_type=baseline_policy) if baseline_policy else None

    if baseline is not None:
        baseline_cumulative = [record["cumulative_cost"] for record in baseline["records"]]
    else:
        baseline_cumulative = []

    records = []
    cumulative_cost = 0.0

    chart_container = st.container()
    metrics_container = st.container()
    progress_container = st.container()
    table_container = st.container()

    with chart_container:
        chart_placeholder = st.empty()
        chart_caption_placeholder = st.empty()

    with metrics_container:
        metrics_placeholder = st.empty()

    with progress_container:
        progress = st.progress(0)
        status = st.empty()

    with table_container:
        with st.expander("Step-by-step details", expanded=False):
            table_placeholder = st.empty()

    for step_index in range(24):
        valid_actions = get_valid_actions(state)
        action = agent.choose_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)

        cumulative_cost += info["cost"]
        state = next_state

        record = {
            "Hour": info["hour"],
            "Action": ACTION_NAMES[action],
            "Step Cost": round(info["cost"], 4),
            "Cumulative Cost": round(cumulative_cost, 4),
            "Battery": round(info["battery"], 2),
            "Grid Import": round(info["grid_import"], 2),
            "Grid Export": round(info["grid_export"], 2),
            "Solar": round(info["solar"], 2),
            "Demand": round(info["demand"], 2),
            "Price": round(float(env.daily_prices[info["hour"]]), 4),
        }
        records.append(record)

        metrics_cols = metrics_placeholder.columns(4)
        metrics_cols[0].metric("Current hour", f"{info['hour']:02d}:00")
        metrics_cols[1].metric("Action", ACTION_NAMES[action])
        metrics_cols[2].metric("Step cost", f"${info['cost']:.2f}")
        metrics_cols[3].metric("Total cost", f"${cumulative_cost:.2f}")

        chart_payload = {
            "Hour": [row["Hour"] for row in records],
            "Agent cumulative cost": [row["Cumulative Cost"] for row in records],
        }
        if baseline_cumulative:
            chart_payload[f"{baseline_choice} baseline"] = baseline_cumulative[: len(records)]

        chart_frame = pd.DataFrame(chart_payload).set_index("Hour")
        chart_placeholder.line_chart(chart_frame, height=260)
        if baseline_cumulative:
            chart_caption_placeholder.caption(
                f"Blue line: trained agent. Orange line: {baseline_choice} baseline on the same synthetic day."
            )
        else:
            chart_caption_placeholder.empty()

        table_placeholder.dataframe(pd.DataFrame(records), use_container_width=True, height=320)
        progress.progress((step_index + 1) / 24)
        status.write(f"Step {step_index + 1}/24 — `{ACTION_NAMES[action]}` on synthetic hour {info['hour']}")

        if speed_ms > 0 and not done:
            time.sleep(speed_ms / 1000.0)

        if done:
            break

    st.success(f"Finished the day. Total cost: ${cumulative_cost:.2f}")
    st.caption("Tip: run the same seed again to compare the exact same synthetic day.")
else:
    st.info("Press **Run live day** to animate the agent across a 24-hour synthetic profile.")
    preview_env = SmartHomeEnv(seed=int(seed))
    preview_env.reset()
    preview_data = pd.DataFrame(
        {
            "Hour": list(range(24)),
            "Price": preview_env.daily_prices,
            "Demand": preview_env.daily_demand,
            "Solar": preview_env.daily_solar,
        }
    )
    st.line_chart(preview_data.set_index("Hour"), height=260)
