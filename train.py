from env.smart_home_env import SmartHomeEnv
from agent.q_agent import QAgent
import numpy as np

# ─────────────────────────────────────────────────────────────
# Valid-action mask
#   0 = charge from grid   → only if battery not full
#   1 = discharge to load  → only if battery not empty
#   2 = idle               → always valid
#   3 = prioritise solar   → only if solar available (solar_bin > 0)
#   4 = sell to grid       → only if battery not empty
# ─────────────────────────────────────────────────────────────
def get_valid_actions(state):
    _hour, battery_bin, _price_bin, _demand_bin, solar_bin = state
    valid = [2]                         # idle is always valid
    if battery_bin < 10: valid.append(0)             # can charge
    if battery_bin > 0:  valid.append(1)             # can discharge
    if solar_bin   > 0:  valid.append(3)             # solar available
    if battery_bin > 0:  valid.append(4)             # can sell
    return valid

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
env    = SmartHomeEnv()
agent  = QAgent()
episodes = 8000
history  = []

print("Training Agent on Financial Rewards...")
print(f"Q-table shape : {agent.shape}  "
      f"({2 * agent.q_table_a.size * 8 / 1024:.0f} KB total for both tables)\n")

for ep in range(episodes):
    state = env.reset()
    agent.reset_traces()
    total_cost = 0.0

    while True:
        valid   = get_valid_actions(state)
        action  = agent.choose_action(state, valid)
        next_state, reward, done, info = env.step(action)

        agent.update(state, action, reward, next_state,
                     get_valid_actions(next_state))

        state      = next_state
        total_cost += info["cost"]
        if done:
            break

    agent.decay_exploration()
    history.append(total_cost)

    if (ep + 1) % 1000 == 0:
        print(f"Episode {ep+1:5d} | "
              f"Avg Cost (last 1k): ${np.mean(history[-1000:]):.2f} | "
              f"ε: {agent.epsilon:.3f}")

# ─────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────
def run_benchmark(policy_type, num_days=200):
    costs = []
    for s in range(num_days):
        test_env = SmartHomeEnv(seed=s + 500)
        state    = test_env.reset()
        day_cost = 0.0

        while True:
            if policy_type == "rl":
                action = agent.choose_action(state, get_valid_actions(state))

            elif policy_type == "tou":
                # Rule: cheap → charge; expensive → discharge; solar → prioritise solar
                _h, bat, price_bin, _d, solar_bin = state
                if solar_bin > 0:
                    action = 3                          # soak up solar
                elif price_bin == 0 and bat < 10:
                    action = 0                          # cheap grid → charge
                elif price_bin >= 3 and bat > 0:
                    action = 1                          # expensive grid → discharge
                else:
                    action = 2                          # idle
                if action not in get_valid_actions(state):
                    action = 2

            else:   # idle
                action = 2

            state, _, done, info = test_env.step(action)
            day_cost += info["cost"]
            if done:
                break

        costs.append(day_cost)
    return np.mean(costs)

print("\nRunning benchmarks (200 days each)...")
rl_avg   = run_benchmark("rl")
tou_avg  = run_benchmark("tou")
idle_avg = run_benchmark("idle")

print(f"\n{'Policy':<10} {'Avg Daily Cost':>15}")
print("-" * 27)
print(f"{'RL':<10} ${rl_avg:>14.2f}")
print(f"{'ToU':<10} ${tou_avg:>14.2f}")
print(f"{'Idle':<10} ${idle_avg:>14.2f}")

improvement_tou  = (tou_avg  - rl_avg) / tou_avg  * 100
improvement_idle = (idle_avg - rl_avg) / idle_avg * 100
print(f"\nRL vs ToU  improvement : {improvement_tou:+.2f}%")
print(f"RL vs Idle improvement : {improvement_idle:+.2f}%")