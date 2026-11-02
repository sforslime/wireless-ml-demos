"""
Synthetic spectrum-occupancy simulator.

A band of C channels is sensed every few seconds. Each channel switches
between idle and busy as a two-state Markov chain whose busy-probability
follows a daily and weekly traffic pattern (busier in the afternoon,
quieter at night and on weekends), plus random day-to-day variation.

We aggregate into 15-minute slots and report the Spectrum Occupancy
Ratio (SOR): fraction of (channel, sensing instant) pairs that were busy.

An optional "surge" (e.g. a big event near the base station) raises
traffic for the last few days, creating a distribution shift.
"""

import numpy as np

SLOTS_PER_DAY = 96          # 15-minute slots
SENSES_PER_SLOT = 30        # sensing instants per slot


def daily_profile(slot_of_day):
    """Typical load shape over one day, in [0, 1]."""
    h = slot_of_day / 4.0                          # hour of day
    morning = np.exp(-((h - 9.5) / 2.0) ** 2)
    afternoon = np.exp(-((h - 15.0) / 2.5) ** 2)
    evening = 0.8 * np.exp(-((h - 20.5) / 1.8) ** 2)
    return 0.08 + 0.55 * np.maximum.reduce([morning, afternoon, evening])


def simulate(n_days=70, n_channels=20, surge_from_day=None, surge_gain=1.6, seed=0):
    rng = np.random.default_rng(seed)
    T = n_days * SLOTS_PER_DAY
    sor = np.zeros(T)
    state = rng.random(n_channels) < 0.2            # True = busy
    chan_bias = rng.uniform(0.6, 1.4, n_channels)   # some channels are busier
    day_level = 1.0

    for t in range(T):
        day, slot = divmod(t, SLOTS_PER_DAY)
        if slot == 0:                                # new day: random daily level
            day_level = rng.lognormal(0, 0.15)
            weekend = (day % 7) in (5, 6)
        load = daily_profile(slot) * day_level * (0.65 if weekend else 1.0)
        if surge_from_day is not None and day >= surge_from_day:
            load *= surge_gain
        p_busy = np.clip(load * chan_bias, 0.01, 0.97)

        # Markov chain with mean busy-burst length ~ 6 sensing instants
        p_end = 1 / 6
        p_start = p_end * p_busy / (1 - p_busy)
        busy_count = 0
        for _ in range(SENSES_PER_SLOT):
            u = rng.random(n_channels)
            state = np.where(state, u > p_end, u < p_start)
            busy_count += state.sum()
        sor[t] = busy_count / (n_channels * SENSES_PER_SLOT)
    return sor
