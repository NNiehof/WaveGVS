import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval


data_folder = "D:/OneDrive/Code/WaveGVS/Data/002/"
data_file = data_folder + "002_waveGVS_phaseshift_20180801_162718.txt"
df = pd.read_csv(data_file, sep="; ", engine="python")
df["final_amp"] = np.nan

# get conditions
currents = df["current"].unique()
frequencies = df["frequency"].unique()
c = len(currents)
f = len(frequencies)
count = 0
x_tick_positions = []
plotlabels = []
fig, ax = plt.subplots()

for curr in currents:
    for freq in frequencies:
        count += 1
        x_tick_positions.append(count)
        # plt.subplot(c, f, count)
        plotlabels.append("{0} mA\n{1} Hz".format(curr, freq))
        selection = df[(df["current"] == curr) & (df["frequency"] == freq)]
        for index, row in selection.iterrows():
            amplitudes = np.array(literal_eval(row["line_amplitude"]))
            # collapse over offset angles
            amplitudes = amplitudes - row["line_offset"]
            samples = np.arange(0, len(amplitudes))

            # get amplitude in final half second (500 samples)
            row["final_amp"] = max(abs(amplitudes[-500::]))
            selection.loc[index, "final_amp"] = max(abs(amplitudes[-500::]))
            line_offset = row["line_offset"]

            # plt.figure(fig1.number)
            # fig1.plot(samples, amplitudes)

            # plt.figure(fig2.number)
            plt.scatter(count, row["final_amp"], color="blue", alpha=0.5)

        # plt.title("{0} mA, {1} Hz".format(curr, freq))
        # plt.ylabel("line angle (deg)")
        # plt.xlabel("t (samples)")

        mean_amp = np.mean(selection["final_amp"])
        plt.scatter(count, mean_amp, color="red", alpha=0.5)

        plt.xticks(x_tick_positions)
        ax.set_xticklabels(plotlabels)
        plt.ylabel("final amplitude (deg)")
        plt.title("Line oscillation amplitude at end of trial")
plt.tight_layout()
plt.show()
