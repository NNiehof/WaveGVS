import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval


data_folder = "D:/OneDrive/Code/WaveGVS/Data/002/"
data_file = data_folder + "002_waveGVS_phaseshift_20180801_162718.txt"
df = pd.read_csv(data_file, sep="; ", engine="python")
print(df.head(10))

# get conditions
currents = df["current"].unique()
frequencies = df["frequency"].unique()
c = len(currents)
f = len(frequencies)
count = 0

for curr in currents:
    for freq in frequencies:
        count += 1
        plt.subplot(c, f, count)
        selection = df[(df["current"] == curr) & (df["frequency"] == freq)]
        for index, row in selection.iterrows():
            amplitudes = np.array(literal_eval(row["line_amplitude"]))
            # collapse over offset angles
            amplitudes = amplitudes - row["line_offset"]
            samples = np.arange(0, len(amplitudes))
            plt.plot(samples, amplitudes)
        plt.title("{0} mA, {1} Hz".format(curr, freq))
        plt.ylabel("line angle (deg)")
        plt.xlabel("t (samples)")

plt.tight_layout()
plt.show()
