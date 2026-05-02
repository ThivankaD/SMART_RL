import numpy as np

class SmartHomeEnv:
    def __init__(self, seed=42):
        self.max_battery         = 10.0
        self.initial_battery     = 5.0
        self.max_charge_rate     = 2.0
        self.max_discharge_rate  = 2.0
        self.charge_efficiency   = 0.95
        self.discharge_efficiency= 0.95
        self.sell_price_factor   = 0.4

        self.base_prices = np.array([
            0.12, 0.11, 0.11, 0.11, 0.12, 0.16,
            0.22, 0.28, 0.34, 0.30, 0.24, 0.20,
            0.18, 0.18, 0.20, 0.24, 0.30, 0.38,
            0.45, 0.42, 0.32, 0.24, 0.18, 0.14,
        ])

        self.base_demand = np.array([
            1.2, 1.1, 1.0, 1.0, 1.2, 1.8,
            2.6, 3.4, 3.8, 3.4, 3.0, 2.8,
            2.6, 2.4, 2.5, 2.8, 3.2, 4.2,
            4.8, 4.5, 3.6, 2.8, 2.2, 1.6,
        ])

        self.base_solar = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.2,
            0.8, 1.6, 2.4, 3.0, 3.2, 3.0,
            2.8, 2.4, 1.8, 1.2, 0.6, 0.2,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])

        self.demand_noise_std = 0.35
        self.solar_noise_std  = 0.25
        self.rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────
    # Daily profile generation (realistic noise)
    # ─────────────────────────────────────────────
    def _generate_daily_profiles(self):
        demand_noise = self.rng.normal(0, self.demand_noise_std, size=24)
        solar_noise  = self.rng.normal(0, self.solar_noise_std,  size=24)
        price_noise  = self.rng.normal(0, 0.02, size=24)

        self.daily_demand = np.clip(self.base_demand + demand_noise, 0.0, None)
        self.daily_solar  = np.clip(self.base_solar  + solar_noise,  0.0, None)
        self.daily_prices = np.clip(self.base_prices + price_noise, 0.08, 0.50)

    # ─────────────────────────────────────────────
    # Discretisation helpers  (compact, stable bins)
    # ─────────────────────────────────────────────
    def _price_bin(self, price):
        # 5 bins: [<0.15, 0.15-0.22, 0.22-0.30, 0.30-0.40, >=0.40]
        return int(np.digitize(price, bins=[0.15, 0.22, 0.30, 0.40]))

    def _demand_bin(self, demand):
        # 4 bins: low(<1.5), medium(1.5-2.8), high(2.8-4.0), peak(>=4.0)
        return int(np.digitize(demand, bins=[1.5, 2.8, 4.0]))

    def _solar_bin(self, solar):
        # 3 bins: none(<0.3), low(0.3-1.8), high(>=1.8)
        return int(np.digitize(solar, bins=[0.3, 1.8]))

    # ─────────────────────────────────────────────
    # State: (hour, battery_bin, price_bin, demand_bin, solar_bin)
    # Shape: 24 × 11 × 5 × 4 × 3  =  15,840 states
    # ─────────────────────────────────────────────
    def _get_state(self):
        battery_bin = int(np.clip(np.round(self.battery), 0, 10))
        return (
            self.hour,
            battery_bin,
            self._price_bin(self.daily_prices[self.hour]),
            self._demand_bin(self.daily_demand[self.hour]),
            self._solar_bin(self.daily_solar[self.hour]),
        )

    def reset(self):
        self.hour    = 0
        self.battery = self.initial_battery
        self._generate_daily_profiles()
        return self._get_state()

    # ─────────────────────────────────────────────
    # Actions
    #   0 = charge from grid
    #   1 = discharge to load
    #   2 = idle
    #   3 = prioritise solar  (aggressive solar storage)
    #   4 = sell to grid      (export battery + surplus solar)
    # ─────────────────────────────────────────────
    def step(self, action):
        if self.hour >= 24:
            raise ValueError("Episode done. Call reset().")

        hour   = self.hour
        demand = float(self.daily_demand[hour])
        solar  = float(self.daily_solar[hour])
        price  = float(self.daily_prices[hour])

        # ── Base solar logic ──────────────────────
        demand_remaining = max(0.0, demand - solar)
        excess_solar     = max(0.0, solar  - demand)

        # Passive solar → battery (always happens first, before action)
        solar_charge = min(excess_solar,
                           self.max_charge_rate,
                           self.max_battery - self.battery)
        self.battery  += solar_charge * self.charge_efficiency
        excess_solar  -= solar_charge

        grid_import   = 0.0
        grid_export   = excess_solar        # default: leftover solar sold passively
        battery_to_load = 0.0
        extra_sell    = 0.0                 # additional battery sold under action 4

        # ── Action execution ──────────────────────
        if action == 0:                     # Charge from grid
            charge_amt = min(self.max_charge_rate,
                             self.max_battery - self.battery)
            self.battery += charge_amt * self.charge_efficiency
            grid_import  += charge_amt

        elif action == 1:                   # Discharge battery → meet home load
            max_needed    = demand_remaining / self.discharge_efficiency
            discharge_amt = min(self.max_discharge_rate,
                                self.battery, max_needed)
            self.battery   -= discharge_amt
            battery_to_load = discharge_amt * self.discharge_efficiency
            demand_remaining -= battery_to_load

        elif action == 2:                   # Idle — do nothing extra
            pass

        elif action == 3:                   # Prioritise solar storage
            # Top-up battery with any solar that passive charging missed
            # (useful when solar > max_charge_rate per step)
            extra_charge = min(excess_solar,
                               self.max_charge_rate,
                               self.max_battery - self.battery)
            self.battery += extra_charge * self.charge_efficiency
            excess_solar -= extra_charge
            grid_export   = excess_solar    # remaining goes to grid

        elif action == 4:                   # Sell to grid
            # Export additional battery capacity at sell price
            extra_sell  = min(self.max_discharge_rate, self.battery)
            self.battery -= extra_sell
            grid_export  += extra_sell      # sold on top of passive solar export

        # ── Grid covers remaining demand ──────────
        grid_import += demand_remaining

        # ── Financial calculation ─────────────────
        total_cost = (grid_import * price) - (grid_export * price * self.sell_price_factor)

        self.battery = float(np.clip(self.battery, 0.0, self.max_battery))
        self.hour   += 1
        done = (self.hour == 24)

        next_state = (
            self._get_state() if not done
            else (0, int(np.round(self.battery)), 0, 0, 0)
        )

        return next_state, -total_cost, done, {
            "cost":        total_cost,
            "hour":        hour,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "solar":       solar,
            "demand":      demand,
            "battery":     self.battery,
        }