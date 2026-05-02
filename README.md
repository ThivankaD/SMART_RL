# Smart Home RL: Demand, Battery, and Time-of-Day Pricing

This project simulates a **24-hour smart home electricity scenario** and trains an **enhanced tabular Q-learning agent** to minimize total daily electricity cost.

## Scenario Components

- **Demand profile**: Household demand varies by hour and includes random daily noise.
- **Solar profile**: Daylight solar production offsets demand and can export excess energy.
- **Battery model**:
	- Capacity-limited storage
	- Charge/discharge rate limits
	- Charge/discharge efficiency losses
- **Time-of-day pricing**: Electricity import price changes each hour.
- **Grid exchange**:
	- Import from grid to satisfy unmet load and optional battery charging
	- Export excess solar at a reduced sell-back factor

## Reinforcement Learning Setup

- **Algorithm**: Double Q-learning with eligibility traces
- **State**: `(hour, battery_bin, net_load_level, price_bin, next_price_bin)`
- **Actions**:
	- `0`: Charge battery from grid
	- `1`: Idle
	- `2`: Discharge battery to supply home
- **Reward**: Negative hourly electricity cost (`reward = -cost`)
- **Objective**: Minimize total electricity cost over 24 hours

## Project Structure

- `env/smart_home_env.py`: Smart home simulation environment
- `agent/q_agent.py`: Enhanced Q-learning agent (double Q, eligibility traces)
- `train.py`: Training loop, evaluation run, and plotting

## Setup (Windows PowerShell)

```powershell
cd "c:\Users\Thiwanka Dissanayaka\Documents\Smart_RL"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Training

```powershell
python train.py
```

During training, the script prints moving average daily cost and epsilon. After training, it prints an evaluation-day hourly table, policy-intelligence metrics, baseline comparison, and a learning-curve plot.