# Smart_RL

`Smart_RL` is a  reinforcement learning project for smart-home energy control.
It trains a tabular Q-learning agent to minimize daily electricity cost in a 24-hour
home environment with battery storage, solar generation, and time-of-use pricing.

## What the environment models

- **Hourly demand**: A synthetic household load profile with daily noise.
- **Hourly solar**: Daytime solar generation that can cover demand or charge the battery.
- **Battery storage**: Limited capacity with charge/discharge limits and efficiency losses.
- **Time-of-use pricing**: Electricity price changes by hour.
- **Grid interaction**: The agent can import from the grid or export excess energy.

## State space

The agent observes a discrete state tuple:

```text
(hour, battery_bin, price_bin, demand_bin, solar_bin)
```

Where:

- `hour`: Current hour of the day, `0..23`
- `battery_bin`: Battery level rounded to an integer bin, `0..10`
- `price_bin`: Discretized electricity price, `0..4`
- `demand_bin`: Discretized household demand, `0..3`
- `solar_bin`: Discretized solar availability, `0..2`

### State meanings

- **`hour`**: Current hour of the day, `0..23`.
- **`battery_bin`**: Battery level rounded to an integer bin, `0..10` (0 = empty, 10 = full).
- **`price_bin`**: Electricity price category (cheap → expensive).
- **`demand_bin`**: Home’s electricity usage for that hour (kWh), grouped into low → peak.
- **`solar_bin`**: Solar power produced for that hour (kWh), grouped into none → high.

Plain-language meanings:

- **Demand**: Home’s electricity usage for that hour (kWh).
- **Solar**: Solar power produced for that hour (kWh).
- **Battery**: Stored energy level in the battery (kWh).
- **Grid import** = how much energy the home buys from the grid that hour (kWh).
- **Grid export** = how much energy the home sells back to the grid (kWh), mostly from excess solar or battery.

## Action space

The agent uses 5 discrete actions:

- `0` = Charge battery from grid
- `1` = Discharge battery to meet home load
- `2` = Idle
- `3` = Prioritize solar charging
- `4` = Sell battery energy to the grid

### Valid-action rules

- `0` is valid only if the battery is not full.
- `1` is valid only if the battery is not empty.
- `2` is always valid.
- `3` is valid only when solar is available.
- `4` is valid only if the battery has energy.

## Reward and objective

- **Reward**: `reward = -cost`
- **Goal**: Minimize total daily electricity cost
- **Episode length**: 24 steps, one per hour

## Learning setup

- **Algorithm**: Tabular Q-learning
- **Training episodes**: `60000`
- **Exploration**: Epsilon-greedy with decay
- **Benchmarking**: RL policy is evaluated greedily after training
- **Extras**: A small heuristic bias is blended into action selection to improve policy quality

## Files

- `env/smart_home_env.py` — Smart-home environment and reward calculation
- `agent/q_agent.py` — Q-learning agent and heuristic action bias
- `smart_rl_runtime.py` — Shared training/simulation helpers and model save/load
- `train.py` — Training loop, benchmark comparison, learning-curve plot, and model export
- `app.py` — Streamlit dashboard for live action-by-action visualization
- `requirements.txt` — Python dependencies

## Setup

### Windows PowerShell

```powershell
cd "c:\Users\Thiwanka Dissanayaka\Documents\Smart_RL"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `python` is not on your PATH, use the virtual environment interpreter directly:

```powershell
& "c:\Users\Thiwanka Dissanayaka\Documents\Smart_RL\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Run training

```powershell
python train.py
```

Or with the venv interpreter:

```powershell
& "c:\Users\Thiwanka Dissanayaka\Documents\Smart_RL\.venv\Scripts\python.exe" train.py
```

After training, the learned policy is saved to `artifacts/q_agent.npz`.

## Run the UI

```powershell
streamlit run app.py
```

Or with the venv interpreter:

```powershell
& "c:\Users\Thiwanka Dissanayaka\Documents\Smart_RL\.venv\Scripts\python.exe" -m streamlit run app.py
```

In the dashboard you can:

- Load the saved Q-agent from `artifacts/q_agent.npz`
- Upload a different `.npz` model if you want to compare runs
- Generate a synthetic 24-hour day with a chosen seed
- Watch each action, battery change, and step cost update live
- Compare the agent’s cumulative cost against an idle baseline on the same day

## What the script prints

After training, `train.py` shows:

- Average training cost every few thousand episodes
- Benchmark comparison for:
  - `RL`
  - `ToU`
  - `Idle`
- Improvement percentages vs the baselines
- A **cost vs episode** plot with a moving average

## Benchmark behavior

The benchmark runs after training is finished.

- **RL**: Uses the learned policy greedily (`epsilon = 0` during evaluation)
- **ToU**: Rule-based time-of-use baseline
- **Idle**: Always chooses idle

## Current learned behavior

The current setup is designed to make the agent learn patterns such as:

- Charge when electricity is cheap
- Discharge during expensive/peak hours
- Use solar when available
- Avoid unnecessary grid imports

## Notes

- The project uses a compact discrete Q-table, so training is fast and easy to inspect.
- The learning curve can be noisy early on; the moving-average plot helps show the trend.
- The benchmark numbers are based on the fixed test seeds in `train.py`.