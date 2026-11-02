"""
Synthetic wireless signal generator.

Each example is a short burst of complex baseband samples (I and Q),
shaped like the samples a software-defined radio would capture.

Impairments added so the task isn't trivial:
  - random carrier phase offset
  - small carrier frequency offset
  - additive white Gaussian noise at a chosen SNR

Output shape per example: (2, N) -> row 0 = I (real), row 1 = Q (imag).
"""

import numpy as np

N_SAMPLES = 128          # samples per example
SPS = 8                  # samples per symbol
N_SYMBOLS = N_SAMPLES // SPS + 8   # extra symbols so filter edges get trimmed


def rrc_taps(beta=0.35, sps=SPS, span=6):
    """Root-raised-cosine pulse-shaping filter (standard in digital comms)."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            taps[i] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(4 * beta * ti) - 1) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps ** 2))


RRC = rrc_taps()


# ---------- constellations ----------
def psk(m):
    return np.exp(1j * (2 * np.pi * np.arange(m) / m + (np.pi / m if m == 4 else 0)))


def qam(m):
    k = int(np.sqrt(m))
    levels = np.arange(-(k - 1), k, 2)
    pts = np.array([a + 1j * b for a in levels for b in levels])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def pam(m):
    pts = np.arange(-(m - 1), m, 2).astype(complex)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    "BPSK": psk(2),
    "QPSK": psk(4),
    "8PSK": psk(8),
    "16QAM": qam(16),
    "64QAM": qam(64),
    "PAM4": pam(4),
}


def linear_mod(name, rng):
    """Pick random symbols, upsample, pulse-shape with RRC."""
    const = CONSTELLATIONS[name]
    syms = rng.choice(const, N_SYMBOLS)
    up = np.zeros(N_SYMBOLS * SPS, dtype=complex)
    up[::SPS] = syms
    shaped = np.convolve(up, RRC, mode="same")
    start = 4 * SPS
    return shaped[start:start + N_SAMPLES]


def fsk(m, rng):
    """Continuous-phase M-FSK."""
    syms = rng.integers(0, m, N_SYMBOLS)
    freqs = (2 * syms - (m - 1)) / (2 * SPS)      # cycles per sample
    inst_freq = np.repeat(freqs, SPS)
    phase = 2 * np.pi * np.cumsum(inst_freq)
    start = 4 * SPS
    return np.exp(1j * phase[start:start + N_SAMPLES])


def am_dsb(rng):
    """Analog AM (double sideband) carrying a random sum of tones."""
    t = np.arange(N_SAMPLES)
    msg = np.zeros(N_SAMPLES)
    for _ in range(3):
        f = rng.uniform(0.005, 0.04)
        msg += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    msg /= np.max(np.abs(msg)) + 1e-9
    return (1 + 0.6 * msg).astype(complex)


def make_clean(name, rng):
    if name in CONSTELLATIONS:
        return linear_mod(name, rng)
    if name == "2FSK":
        return fsk(2, rng)
    if name == "4FSK":
        return fsk(4, rng)
    if name == "AM-DSB":
        return am_dsb(rng)
    raise ValueError(name)


def impair(x, snr_db, rng):
    """Normalize power, add phase/frequency offset and AWGN."""
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    phase = rng.uniform(0, 2 * np.pi)
    cfo = rng.uniform(-0.01, 0.01)               # cycles per sample
    n = np.arange(len(x))
    x = x * np.exp(1j * (phase + 2 * np.pi * cfo * n))
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise


def make_dataset(classes, snrs, n_per, seed=0):
    """Return X (num, 2, N), y (class index into `classes`), snr (dB)."""
    rng = np.random.default_rng(seed)
    X, y, s = [], [], []
    for ci, name in enumerate(classes):
        for snr in snrs:
            for _ in range(n_per):
                sig = impair(make_clean(name, rng), snr, rng)
                X.append(np.stack([sig.real, sig.imag]))
                y.append(ci)
                s.append(snr)
    return np.array(X, dtype=np.float32), np.array(y), np.array(s)
