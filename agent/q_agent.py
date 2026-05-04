import numpy as np
from pathlib import Path

class QAgent:
    def __init__(
        self,
        alpha=0.10,
        gamma=1.00,  # finite-horizon uniform-cost task
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99985,
    ):
        # ─────────────────────────────────────────────────────────────
        # State shape: (hour:24, battery:11, price:5, demand:4, solar:3)
        # Action dims: 5
        # Total Q-entries per table: 24×11×5×4×3×5 = 79 200
        # ─────────────────────────────────────────────────────────────
        self.shape = (24, 11, 5, 4, 3, 5)

        self.q_table = np.zeros(self.shape)
        self._seed_heuristic_prior()

        self.alpha        = alpha
        self.gamma        = gamma

        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay

    def _seed_heuristic_prior(self):
        price_bonus = np.array([0.05, 0.03, 0.00, 0.04, 0.01], dtype=float)

        for hour in range(24):
            for battery in range(11):
                for price_bin in range(5):
                    for demand_bin in range(4):
                        for solar_bin in range(3):
                            idx = (hour, battery, price_bin, demand_bin, solar_bin)

                            self.q_table[idx + (2,)] = 0.00

                            if battery < 10 and price_bin <= 1:
                                self.q_table[idx + (0,)] = 0.08 + price_bonus[price_bin]

                            if battery > 0 and price_bin >= 3:
                                self.q_table[idx + (1,)] = 0.12 + price_bonus[price_bin]
                                self.q_table[idx + (4,)] = 0.06 + price_bonus[price_bin] * 0.5

                            if solar_bin > 0:
                                self.q_table[idx + (3,)] = 0.10 + solar_bin * 0.03

    def _heuristic_bias(self, state):
        hour, battery, price_bin, demand_bin, solar_bin = state
        bias = np.zeros(5, dtype=float)

        cheap_hours = hour <= 5 or hour >= 21
        peak_hours = 17 <= hour <= 20
        solar_hours = 6 <= hour <= 17

        if battery < 10 and (cheap_hours or price_bin <= 1):
            bias[0] += 0.10

        if battery > 0 and (peak_hours or price_bin >= 3):
            bias[1] += 0.16
            bias[4] += 0.06

        if solar_bin > 0 and solar_hours:
            bias[3] += 0.18 + 0.02 * solar_bin

        if demand_bin >= 3 and battery > 0:
            bias[1] += 0.05

        if not cheap_hours and price_bin >= 2 and battery >= 9:
            bias[2] += 0.03

        return bias

    # ─────────────────────────────────────────────
    # State indexing  (5-tuple → clipped indices)
    # ─────────────────────────────────────────────
    def _state_index(self, state):
        h, b, p, d, s = state
        return (
            int(np.clip(h, 0, 23)),
            int(np.clip(b, 0, 10)),
            int(np.clip(p, 0,  4)),
            int(np.clip(d, 0,  3)),
            int(np.clip(s, 0,  2)),
        )

    # ─────────────────────────────────────────────
    # Action selection (epsilon-greedy, valid mask)
    # ─────────────────────────────────────────────
    def choose_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3, 4]

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        idx = self._state_index(state)
        combined = self.q_table[idx] + self._heuristic_bias(state)

        masked = np.full(5, -np.inf)
        for a in valid_actions:
            masked[a] = combined[a]

        return int(np.argmax(masked))

    # ─────────────────────────────────────────────
    # Tabular Q-learning
    # ─────────────────────────────────────────────
    def update(self, state, action, reward, next_state, next_valid_actions=None, done=False):
        s      = self._state_index(state)
        s_next = self._state_index(next_state)

        if next_valid_actions is None:
            next_valid_actions = [0, 1, 2, 3, 4]

        if done:
            td_target = reward
        else:
            next_q = self.q_table[s_next]
            masked_next = np.full(5, -np.inf)
            for a in next_valid_actions:
                masked_next[a] = next_q[a]
            td_target = reward + self.gamma * np.max(masked_next)

        td_error = td_target - self.q_table[s + (action,)]
        self.q_table[s + (action,)] += self.alpha * td_error

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def save(self, file_path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q_table=self.q_table,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
        )

    @classmethod
    def load(cls, file_path):
        data = np.load(file_path, allow_pickle=False)
        agent = cls(
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
            epsilon=float(data["epsilon"]),
            epsilon_min=float(data["epsilon_min"]),
            epsilon_decay=float(data["epsilon_decay"]),
        )
        agent.q_table = data["q_table"]
        return agent