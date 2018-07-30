from random import shuffle
import numpy as np


class RandStim:

    def __init__(self, current_mA=None, frequency=None, line_offset=None,
                 line_amplitude=None):
        """
        Object that creates a randomised list of trials, out of a range of
        stimulus values and conditions.
        """
        self.trial_list = []
        for curr in current_mA:
            for freq in frequency:
                for line_mu in line_offset:
                    for line_amp in line_amplitude:
                        self.trial_list.append([curr, freq, line_mu, line_amp])
        shuffle(self.trial_list)

    def get_stimulus(self, trial_nr):
        """
        Return stimulus and conditions for next trial
        :return trial: list with conditions
        """
        return self.trial_list[trial_nr]

    def get_n_trials(self):
        """
        Return number of trials in trial list.
        """
        return len(self.trial_list)


if __name__ == "__main__":
    cond = {"current_mA": [0.5, 3.0, 1.5], "frequency": [5.0, 2.5, 1.0],
            "line_offset": [1.0, 20.0, 30.0], "line_amplitude": [10.0, 5.0]}
    s = RandStim(**cond)
    print(s.trial_list)
    print(np.shape(s.trial_list))
