# Nynke Niehof, 2018

import numpy as np
from Experiment.GVS import GVS
from Experiment.GenStim import GenStim


def habituation_signal():
    """
    Generate a habituation signal with a slow ramp
    """
    amp = 2.0
    duration = 25.0
    f_samp = 1e3
    frequency = 1.0
    buffer_size = int(duration * f_samp)
    gvs = GVS(max_voltage=amp)
    timing = {"rate": f_samp, "samps_per_chan": buffer_size}
    connected = gvs.connect("cDAQ1Mod1/ao0", **timing)
    if connected:
        # sine wave
        make_stim = GenStim(f_samp=f_samp)
        make_stim.sine(duration, amp, frequency)
        make_stim.fade(f_samp * 10.0)
        gvs_wave = make_stim.stim

        print("start galvanic stim")
        gvs.write_to_channel(gvs_wave)
        print("end galvanic stim")
        gvs.quit()


if __name__ == "__main__":
    habituation_signal()

