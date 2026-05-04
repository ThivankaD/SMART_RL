import matplotlib.pyplot as plt
import numpy as np

from smart_rl_runtime import (
    load_agent,
    run_benchmark,
    save_agent,
    train_agent,
)


def main():
    episodes = 60000

    print("Training Agent on Financial Rewards...")
    print(f"Total episodes: {episodes}\n")
    agent, history = train_agent(episodes=episodes)
    print(f"Q-table shape : {agent.shape}  ({agent.q_table.size * 8 / 1024:.0f} KB total)\n")

    model_path = save_agent(agent)
    print(f"Saved trained agent to {model_path}")

    print("\nRunning benchmarks (200 days each)...")
    rl_avg = run_benchmark(agent, "rl")
    tou_avg = run_benchmark(agent, "tou")
    idle_avg = run_benchmark(agent, "idle")

    print(f"\n{'Policy':<10} {'Avg Daily Cost':>15}")
    print("-" * 27)
    print(f"{'RL':<10} ${rl_avg:>14.2f}")
    print(f"{'ToU':<10} ${tou_avg:>14.2f}")
    print(f"{'Idle':<10} ${idle_avg:>14.2f}")

    improvement_tou = (tou_avg - rl_avg) / tou_avg * 100
    improvement_idle = (idle_avg - rl_avg) / idle_avg * 100
    print(f"\nRL vs ToU  improvement : {improvement_tou:+.2f}%")
    print(f"RL vs Idle improvement : {improvement_idle:+.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history, linewidth=1.2, label="Episode Cost")

    if len(history) >= 100:
        rolling_window = 100
        rolling_avg = np.convolve(history, np.ones(rolling_window) / rolling_window, mode="valid")
        plt.plot(
            range(rolling_window, len(history) + 1),
            rolling_avg,
            linewidth=2.0,
            label=f"{rolling_window}-Episode Moving Avg",
        )

    plt.title("Training Cost vs Episode")
    plt.xlabel("Episode")
    plt.ylabel("Daily Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()