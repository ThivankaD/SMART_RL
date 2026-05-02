import numpy as np

class QAgent:
    def __init__(
        self,
        alpha=0.10,
        gamma=0.99,
        trace_lambda=0.80,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9992,   # tuned for 8 000 episodes
    ):
        # ─────────────────────────────────────────────────────────────
        # State shape: (hour:24, battery:11, price:5, demand:4, solar:3)
        # Action dims: 5
        # Total Q-entries per table: 24×11×5×4×3×5 = 79 200  ≈ 0.6 MB float64
        # Two tables (Double Q): ~1.2 MB  — memory-efficient ✓
        # ─────────────────────────────────────────────────────────────
        self.shape = (24, 11, 5, 4, 3, 5)

        self.q_table_a = np.zeros(self.shape)
        self.q_table_b = np.zeros(self.shape)

        self.trace_a = np.zeros(self.shape)
        self.trace_b = np.zeros(self.shape)

        self.alpha        = alpha
        self.gamma        = gamma
        self.trace_lambda = trace_lambda

        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay

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
        combined = self.q_table_a[idx] + self.q_table_b[idx]   # shape (5,)

        masked = np.full(5, -np.inf)
        for a in valid_actions:
            masked[a] = combined[a]

        return int(np.argmax(masked))

    def reset_traces(self):
        self.trace_a.fill(0.0)
        self.trace_b.fill(0.0)

    # ─────────────────────────────────────────────
    # Double Q-learning + Eligibility Traces
    # ─────────────────────────────────────────────
    def update(self, state, action, reward, next_state, next_valid_actions=None):
        s      = self._state_index(state)
        s_next = self._state_index(next_state)

        if next_valid_actions is None:
            next_valid_actions = [0, 1, 2, 3, 4]

        # Randomly assign update/eval roles (Double Q)
        if np.random.rand() < 0.5:
            q_target, q_eval = self.q_table_a, self.q_table_b
            traces           = self.trace_a
        else:
            q_target, q_eval = self.q_table_b, self.q_table_a
            traces           = self.trace_b

        # Best next action from target; evaluate with eval table
        best_next_a = next_valid_actions[0]
        max_val     = -np.inf
        for a in next_valid_actions:
            val = q_target[s_next + (a,)]
            if val > max_val:
                max_val     = val
                best_next_a = a

        td_target = reward + self.gamma * q_eval[s_next + (best_next_a,)]
        td_error  = td_target - q_target[s + (action,)]

        # Eligibility trace (replacing-style via accumulate then clip)
        traces *= self.gamma * self.trace_lambda
        traces[s + (action,)] += 1.0

        q_target += self.alpha * td_error * traces

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)