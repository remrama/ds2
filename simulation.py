import numpy as np

def simulate_eeg(time, components, noise_level=20):
    """
    Simulate EEG data with customizable frequency and amplitude components.

    Parameters:
    - time: array of time points
    - components: list of tuples (frequency, amplitude) for EEG components
    - noise_level: standard deviation of the Gaussian noise
    """
    eeg_signal = np.zeros_like(time)
    for freq, amp in components:
        eeg_signal += amp * np.sin(2 * np.pi * freq * time)

    # Add Gaussian noise
    eeg_signal += np.random.normal(0, noise_level, time.shape)
    
    return eeg_signal

def simulate_eog(time, state="wake", noise_level=5, saccade_count=10, saccade_amplitude=100):
    """
    Simulate EOG data for LOC and ROC with customizable parameters.

    Parameters:
    - time: array of time points
    - state: 'wake' or 'rem' for wakefulness or REM sleep, respectively
    - noise_level: standard deviation of the Gaussian noise for the baseline
    - saccade_count: number of saccades (eye movements)
    - saccade_amplitude: amplitude of saccades
    """
    eog_signal = np.random.normal(0, noise_level, time.shape)

    # Add saccades for wakefulness
    if state == 'wake':
        for _ in range(saccade_count):
            saccade_start = np.random.randint(0, len(time))
            eog_signal[saccade_start] += np.random.choice([-1, 1]) * saccade_amplitude
    elif state == 'rem':
        # For REM sleep, simulate rapid eye movements
        rem_movements = np.cumsum(np.random.normal(0, 0.5, time.shape))  # Random walk for eye movements
        eog_signal += rem_movements

    return eog_signal

def simulate_emg(time, noise_level=1, burst_count=10, burst_amplitude=3, burst_duration_samples=50):
    """
    Simulate EMG data with customizable parameters.

    Parameters:
    - time: array of time points
    - noise_level: standard deviation of the Gaussian noise for the baseline
    - burst_count: number of muscle activity bursts
    - burst_amplitude: amplitude of muscle bursts
    - burst_duration_samples: duration of muscle bursts in samples
    """
    # Start with baseline noise
    emg_signal = np.random.normal(0, noise_level, time.shape)

    # Add bursts of muscle activity
    for _ in range(burst_count):
        start_index = np.random.randint(0, len(time) - burst_duration_samples)
        burst = np.random.normal(0, burst_amplitude, burst_duration_samples)
        emg_signal[start_index:start_index + burst_duration_samples] = burst

    return emg_signal
