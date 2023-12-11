import argparse
import matplotlib.pyplot as plt
import numpy as np

from simulation import simulate_eeg, simulate_eog, simulate_emg

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=32, help="Random seed")
parser.add_argument("--state", type=str, default="rem", choices=["wec", "rem"])
args = parser.parse_args()

seed = args.seed
state = args.state

# Set the random seed for reproducibility
np.random.seed(seed=seed)

# Simulation parameters
frequency = 256  # Sampling frequency (Hz)
duration = 15  # Duration (seconds)
time = np.arange(0, duration, 1/frequency)


if state == "rem":
    eeg_kwargs = {"noise_level": 15}
    emg_kwargs = {
        "noise_level": 0.2,
        "burst_count": 0,
        "burst_amplitude": 0.5,
        "burst_duration_samples": 25,
    }
    eog_kwargs = {
        "state": "rem",
        "noise_level": 0, #5,
    }
    eeg_components = [
        (10, 10),   # Alpha wave: 10 Hz, 30 µV
        (6, 10),    # Theta wave: 6 Hz, 30 µV
        (20, 6)    # Beta wave: 20 Hz, 15 µV
    ]
elif state == "wec":
    eeg_kwargs = {"noise_level": 15}
    emg_kwargs = {
        "noise_level": 2,
        "burst_count": 10,
        "burst_amplitude": 3,
        "burst_duration_samples": 50,
    }
    eog_kwargs = {
        "state": "wake",
        "noise_level": 1, #5,
        "saccade_count": 0, #10,
        "saccade_amplitude": 100,
    }
    eeg_components = [
        (10, 40),  # Alpha wave: 10 Hz, 100 µV
        (6, 10),    # Theta wave: 6 Hz, 20 µV
        (20, 5)    # Beta wave: 20 Hz, 10 µV
    ]

eeg = simulate_eeg(time, eeg_components, **eeg_kwargs)
emg = simulate_emg(time, **emg_kwargs)
eog = simulate_eog(time, **eog_kwargs)

# Add LRLR signal to EOG
event_start = 9  # time into the main signal this will show up
event_duration = 2  # seconds
event_amplitude = 40  # µV
event_frequency = 1  # Hz
event_time = np.arange(0, event_duration, 1/frequency)
# Generate the event signal (2-cycle sinusoid within the event window)
event_signal = event_amplitude * np.sin(2 * np.pi * event_frequency * event_time)
# Determine the indices of the event start and end times within the original time array
event_end = event_start + event_duration
start_index = int(event_start * frequency)
end_index = int(event_end * frequency)
# # Create a new array to hold the combined signal
# combined_signal = np.copy(wake_loc)
# Add the event signal to the combined signal at the correct indices
eog[start_index:end_index] += event_signal


if state == "rem":
    from scipy.signal import butter, filtfilt
    cutoff = 0.1  # The cutoff frequency of the high-pass filter in Hz
    # Create a high-pass Butterworth filter
    b, a = butter(N=2, Wn=cutoff / (0.5 * frequency), btype="high", analog=False)
    # Apply the filter to the signal
    eog = filtfilt(b, a, eog)


data_and_labels = {
    "EEG": eeg,
    "EMG": emg,
    "EOG": eog,
}

labels, data = zip(*data_and_labels.items())

figsize = (3, 2)
ybound = 80
nrows = len(data_and_labels)
plot_kwargs = {"linewidth": 0.5, "color": "black"}

fig, axes = plt.subplots(
    nrows=nrows, figsize=figsize,
    sharey=False, sharex=True,
    constrained_layout=True,
    gridspec_kw={"hspace": 0, "height_ratios": [1, 1, 1.5]},
)

for i, ax in enumerate(axes):
    ax.plot(time, data[i], **plot_kwargs)
    if labels[i] == "EOG":
        ax.plot(time, -1 * data[i], **plot_kwargs)
    ax.set_yticks([0])
    ax.set_yticklabels([labels[i]], fontsize=10)

# Add a scale bar
y_scale = 50  # For example, 1 µV
x_scale = 2  # For example, 1 second
yy = 2
ax.plot([0, 0], [0-ybound*yy, y_scale-ybound*yy], color="black", linewidth=1, clip_on=False)
ax.plot([0, x_scale], [0-ybound*yy, 0-ybound*yy], color="black", linewidth=1, clip_on=False)
ax.text(x_scale/2, 0-ybound*yy, f'{x_scale} s', verticalalignment='top', horizontalalignment='center', clip_on=False, fontsize=6)
ax.text(0, (y_scale-ybound*yy)-(y_scale/2), f'{y_scale} µV', verticalalignment='center', horizontalalignment='right', clip_on=False, fontsize=6)

for ax in axes:
    # ax.set_aspect("equal")
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.spines["right"].set_position(("outward", 5))
    ax.margins(x=0)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False)
    ax.set_ylim(-ybound, ybound)


plt.savefig(f"../output/figure-3a_{state}.png", dpi=300)
plt.savefig(f"../output/figure-3a_{state}.svg", format="svg")
plt.savefig(f"../output/figure-3a_{state}.eps", format="eps")
plt.savefig(f"../output/figure-3a_{state}.pdf", format="pdf")
plt.close()
