# Nynke Niehof, 2018

from __future__ import division

import numpy as np


class GenStim(object):
    def __init__(self, f_samp=1e3):
        """
        Class to create single-channel stimulus arrays.

        :param f_samp: sampling frequency (Hz)
        """

        self.f_samp = f_samp
        self.n_samp = 0
        self.stim = np.empty(shape=(0, 0))
        self.fade_samples = 0

    def noise(self, duration, amp, bandwidth=(-np.inf, np.inf)):
        """
        Generate noise from random samples on the interval
        Unif[-amplitude, amplitude)

        :param duration: signal duration (seconds)
        :param amp: maximum signal amplitude
        :param bandwidth: (tuple) bandwidth limits (Hz)
        :return:
        """
        self.n_samp = int(duration * self.f_samp)
        self.stim = (2 * amp) * np.random.random(size=self.n_samp) - amp

    def sine(self, duration, amp, frequency):
        self.n_samp = int(duration * self.f_samp)
        t = np.arange(0, duration, 1.0 / self.f_samp)
        self.stim = amp * np.sin(2 * np.pi * frequency * t)

    def fade(self, fade_samples):
        """
        Fade genStim.stim in and out for a duration given by *fade_samples*.

        :param fade_samples: length of the fade in samples (will be rounded
        off to the nearest integer)
        """
        self.fade_samples = int(round(fade_samples))
        fader = np.ones(self.n_samp)
        samp = np.arange(0, self.fade_samples)
        ramp = np.square(np.sin(0.5 * np.pi * samp / self.fade_samples))
        # fade in
        fader[0:self.fade_samples] = ramp
        # fade out
        fader[(len(fader) - self.fade_samples):] = ramp[::-1]
        # apply the fades
        self.stim = self.stim * fader
