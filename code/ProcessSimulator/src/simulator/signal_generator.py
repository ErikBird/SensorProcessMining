import math
import random
import operator
import numpy as np

## Anomalies:

def add_trend(x, strength=0.1, positiv=False, negative=False):
    if not positiv or negative:
        positiv = np.random.choice([True, False])
    time = np.arange(len(x))
    if positiv:
        trend = time * ((max(x) * strength) / len(x))
    else:
        trend = time * -((max(x) * strength) / len(x))
    return x + trend



def add_noise(x, strength=0.1):
    mean_noise = 0
    variance = max(max(x) - min(x), 1)
    noise = np.random.normal(mean_noise, variance * strength, len(x))
    return x + noise


def add_spike(x):
    random_index = random.randint(0, len(x) - 1)
    amplitude = random.uniform(5.0, 5.5)
    if np.random.choice([False, True], p=[0.5, 0.5]):
        x[random_index] = x[random_index] * amplitude
    else:
        x[random_index] = x[random_index] / amplitude
    return x


def add_increase_by_percent(x, percent=0.1):
    constant = max(x) * percent
    return [value + constant for value in x]


def add_spike_cluster(x):
    start_index = random.randint(0, int(len(x) * 0.95))
    size = np.random.uniform(low=0.01, high=0.05)
    increase = np.random.choice([True, False], p=[0.50, 0.50])
    end_index = int(len(x) * size) + start_index
    for index in range(start_index, end_index):
        factor = np.random.uniform(low=0.2, high=0.5)
        if increase: factor = (1 - factor) + 1
        x[index] = x[index] * factor
    return x


def add_stronger_peaks(x, factor=0.1, top_p=0.1):
    # Increases only values in the highest range
    increase_from = max(x) - ((max(x) - min(x)) * top_p)
    for i in range(len(x)):
        if x[i] > increase_from:
            x[i] = x[i] * (1 + factor)
    return x


def add_stuck_at_zero(x):
    start_index = random.randint(0, int(len(x) / 2))
    end_index = random.randint(int(len(x) / 2), len(x) - 1)
    for index in range(start_index, end_index):
        x[index] = 0
    return x


def add_stuck_at_constant(x):
    start_index = random.randint(0, int(len(x) / 2))
    end_index = random.randint(int(len(x) / 2), len(x) - 1)
    constant_value = x[start_index]
    for index in range(start_index, end_index):
        x[index] = constant_value
    return x


## Helpers:
def padding(inputs, size):
    result = np.zeros(size)
    result[:len(inputs)] = inputs
    return result


def transition(inputs, position='start', strength=2):
    if position == 'start':
        for i, x in enumerate(inputs):
            inputs[i] = x * min(np.log(i + 1) / strength, 1)
    elif position == 'stop':
        for i, x in enumerate(reversed(inputs)):
            inputs[-i - 1] = x * min(np.log(i + 1) / strength, 1)
    elif position == 'bilateral':
        for i, x in enumerate(reversed(inputs)):
            inputs[i] = x * min(np.log(i + 1) / strength, 1)
            inputs[-i - 1] = x * min(np.log(i + 1) / strength, 1)
    return inputs

## Signals:
def spaced_sawtooth(periods, amplitude, anomaly=False):
    def generate_sawtooth(amplitude, sawtooth_length=10, empty_length=10):
        data = []
        period_length = sawtooth_length + empty_length
        for i in range(period_length):
            if i % period_length in range(sawtooth_length):
                result = (i % period_length) * (amplitude / sawtooth_length)
                data.append(result)
            else:
                data.append(0)
        return data

    output = []
    anomaly_period = None
    if anomaly:
        anomaly_period = random.randint(0, periods - 1)
    for period in range(periods):
        if period == anomaly_period:
            multiplier = random.uniform(1.5, 3)
            output = output + generate_sawtooth(amplitude=amplitude * multiplier)
        else:
            output = output + generate_sawtooth(amplitude=amplitude)
    return output


def perlin_noise(amplitude=1, wavelength=2000, octaves=5, divisor=2, num_points=200, fixed=20, variable=5):
    # Cosine Interpolation
    def cerp(y_0, y_1, alpha):
        ft = alpha * math.pi
        f = (1 - math.cos(ft * math.pi)) / 2
        return (1 - f) * y_0 + f * y_1

    def noise(amp, wl, a, b):
        out = []
        x = 0
        for _ in range(num_points):
            if x % wl == 0:
                a = b
                b = random.random()
                y = a * amp
            else:
                y = cerp(a, b, x / wl) * amp
                # Pattern repeats with:
                # cerp(a,b, (x%wl) / wl)* amp
            out.append(y)
            x = x + 1
        return out

    signal_out = None
    amp_tmp, wl_tmp = amplitude, wavelength
    for _ in range(1, octaves + 1):
        signal_temp = noise(amp=amp_tmp, wl=wl_tmp, a=random.random(), b=random.random())
        if not signal_out:
            signal_out = signal_temp
        else:
            signal_out = list(map(operator.add, signal_out, signal_temp))
        amp_tmp /= divisor
        wl_tmp /= divisor
    signal_normalized = [((float(i) - min(signal_out)) / (max(signal_out) - min(signal_out))) * variable for i in
                         signal_out]
    signal_total = [x + fixed for x in signal_normalized]
    return signal_total
