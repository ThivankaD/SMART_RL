from pathlib import Path

import numpy as np

from agent.q_agent import QAgent
from env.smart_home_env import SmartHomeEnv

ACTION_NAMES = {
    0: "Charge from grid",
    1: "Discharge to load",
    2: "Idle",
    3: "Prioritize solar",
    4: "Sell to grid",
}


def get_valid_actions(state):
    _hour, battery_bin, _price_bin, _demand_bin, solar_bin = state
    valid = [2]
    if battery_bin < 10:
        valid.append(0)
    if battery_bin > 0:
        valid.append(1)
    if solar_bin > 0:
        valid.append(3)
    if battery_bin > 0:
        valid.append(4)
    return valid


def choose_rule_based_action(state):
    _hour, battery_bin, price_bin, _demand_bin, solar_bin = state

    if solar_bin > 0:
        action = 3
    elif price_bin == 0 and battery_bin < 10:
        action = 0
    elif price_bin >= 3 and battery_bin > 0:
        action = 1
    else:
        action = 2

    return action if action in get_valid_actions(state) else 2


def run_episode(agent, seed=42, policy_type="rl", greedy=False):
    env = SmartHomeEnv(seed=seed)
    state = env.reset()
    step_records = []
    total_cost = 0.0

    while True:
        valid_actions = get_valid_actions(state)

        if policy_type == "rl":
            original_epsilon = agent.epsilon
            if greedy:
                agent.epsilon = 0.0
            action = agent.choose_action(state, valid_actions)
            agent.epsilon = original_epsilon
        elif policy_type == "tou":
            action = choose_rule_based_action(state)
        else:
            action = 2

        next_state, _reward, done, info = env.step(action)
        total_cost += info["cost"]

        step_records.append(
            {
                "hour": info["hour"],
                "state": state,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "cost": info["cost"],
                "cumulative_cost": total_cost,
                "battery": info["battery"],
                "grid_import": info["grid_import"],
                "grid_export": info["grid_export"],
                "solar": info["solar"],
                "demand": info["demand"],
                "price": float(env.daily_prices[info["hour"]]),
            }
        )

        state = next_state
        if done:
            break

    return {
        "records": step_records,
        "total_cost": total_cost,
        "prices": env.daily_prices.tolist(),
        "demand": env.daily_demand.tolist(),
        "solar": env.daily_solar.tolist(),
    }


def run_benchmark(agent, policy_type, num_days=200):
    saved_epsilon = agent.epsilon
    costs = []

    try:
        if policy_type == "rl":
            agent.epsilon = 0.0

        for seed in range(500, 500 + num_days):
            result = run_episode(agent, seed=seed, policy_type=policy_type, greedy=True)
            costs.append(result["total_cost"])
    finally:
        agent.epsilon = saved_epsilon

    return float(np.mean(costs))


def train_agent(episodes=60000, seed=42):
    env = SmartHomeEnv(seed=seed)
    agent = QAgent()
    history = []

    for episode_num in range(episodes):
        state = env.reset()
        total_cost = 0.0

        while True:
            valid = get_valid_actions(state)
            action = agent.choose_action(state, valid)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, get_valid_actions(next_state), done=done)

            state = next_state
            total_cost += info["cost"]
            if done:
                break

        agent.decay_exploration()
        history.append(total_cost)

        if (episode_num + 1) % 4000 == 0:
            avg_cost = np.mean(history[-4000:])
            print(f"Episode {episode_num + 1:5d} | Avg Cost (last 4k): ${avg_cost:8.2f} | ε: {agent.epsilon:.3f}")

    return agent, history


def save_agent(agent, model_path="artifacts/q_agent.npz"):
    path = Path(model_path)
    agent.save(path)
    return path


def load_agent(model_path="artifacts/q_agent.npz"):
    return QAgent.load(model_path)
