import numpy as np
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
import random

def generate_random_clean_waveforms(num_samples, sample_rate=4096):
    signals = []

    for _ in range(num_samples):
        mass1 = random.uniform(10, 50)
        mass2 = random.uniform(10, 50)
        f_lower = 20

        hp, _ = get_td_waveform(
            approximant="IMRPhenomD",
            mass1=mass1,
            mass2=mass2,
            delta_t=1.0/sample_rate,
            f_lower=f_lower
        )

        sig = TimeSeries(hp, delta_t=1.0/sample_rate)

        sig.resize(4096)

        arr = np.array(sig)
        arr = arr / np.max(np.abs(arr) + 1e-12)
        signals.append(arr)

    return np.array(signals)

signals = generate_random_clean_waveforms(5000)
np.save('../../data/raw/clean_signals.npy', signals)
print("✅ Saved 5000 randomized clean signals!")
