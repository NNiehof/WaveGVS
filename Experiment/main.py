import os
import json
from collections import OrderedDict
import numpy as np
from psychopy import visual, core, event
from Experiment.GVS import GVS

"""
Present sinusoidal GVS with a visual line oscillating around the
occipito-nasal axis at the same sine frequency. Participants can set the
amplitude of the visual line to match their subjective visual vertical, so
that they see the line as standing still and upright.
"""


class WaveExp:

    def __init__(self):

        # constants
        self.f_sampling = 1e3
        self.screen_refresh_freq = 60
        self.n_trials = 2
        self.duration_s = 10.0
        self.current_mA = 1.0
        self.stochastic_gvs = False
        self.physical_channel_name = "cDAQ1Mod1/ao0"
        self.line_amplitude_step_size = 1.0

        # TODO: generate trial list
        # frequency
        self.trial_list = [[0.5, 1.0], [1.0, 1.0]]

        # initialise
        self.make_stim = None
        self.stimuli = None
        self.triggers = None
        self.gvs_wave = None
        self.visual_wave = None
        self.line_amplitude = 1.0

        # root directory
        abs_path = os.path.abspath("__file__")
        self.root_dir = os.path.dirname(os.path.dirname(abs_path))
        self.settings_dir = "{}/Settings".format(self.root_dir)

    def setup(self):
        # display and window settings
        self._display_setup()

        # set up connection with galvanic stimulator
        self._gvs_setup()

        # create stimuli
        self.make_stim = Stimuli(self.win, self.settings_dir, self.n_trials)
        self.stimuli, self.triggers = self.make_stim.create()

    def _display_setup(self):
        """
        Window and display settings
        """
        display_file = "{}/display.json".format(self.settings_dir)
        with open(display_file) as json_file:
            win_settings = json.load(json_file)
        self.win = visual.Window(**win_settings)
        self.mouse = event.Mouse(visible=False, win=self.win)

    def _gvs_setup(self):
        """
        Establish connection with galvanic stimulator
        """
        buffer_size = int(self.duration_s * self.f_sampling)
        timing = {"rate": self.f_sampling, "samps_per_chan": buffer_size}
        self.gvs = GVS()
        self.gvs.connect(self.physical_channel_name, **timing)

    def make_waves(self, frequency, line_amplitude):
        """
        Make sine GVS signal and and cosine visual line angle of the
        given frequency.
        :return: gvs_wave, line_angle
        """
        gvs_time = np.arange(0, self.duration_s,
                             1.0 / self.f_sampling)
        gvs_wave = self.current_mA * np.sin(2 * np.pi * frequency * gvs_time)
        visual_time = np.arange(0, self.duration_s,
                                1.0 / (self.duration_s * self.screen_refresh_freq))
        visual_wave = line_amplitude * np.cos(2 * np.pi * frequency * visual_time)
        return gvs_wave, visual_wave

    def check_response(self):
        """
        Check for key presses, update the visual line amplitude
        """
        key_response = event.getKeys(keyList=["left", "right", "escape"])
        if key_response:
            if "left" in key_response:
                self.line_amplitude -= self.line_amplitude_step_size
            elif "right" in key_response:
                self.line_amplitude += self.line_amplitude_step_size
            elif "escape" in key_response:
                self.quit_exp()

    def display_stimuli(self):
        """
        Draw stimuli on screen
        """
        for stim in self.stimuli:
            if self.triggers[stim]:
                self.stimuli[stim].draw()
        self.win.flip()

    def run(self):
        """
        Run the experiment
        """
        for trial in self.trial_list:
            frequency = trial[0]
            self.line_amplitude = trial[1]
            self.gvs_wave, self.visual_wave = self.make_waves(
                frequency, self.line_amplitude)

            # send GVS signal
            self.stimulus_plot(self.gvs_wave)
            self.gvs.write_to_channel(self.gvs_wave,
                                      reset_to_zero_volts=True)
            self.triggers["rodStim"] = True

            # draw visual line
            for frame in self.visual_wave:
                self.stimuli["rodStim"].ori = frame * self.line_amplitude
                self.display_stimuli()
                self.check_response()

            self.triggers["rodStim"] = False

    def stimulus_plot(self, stim, title=""):
        """
        Plot generated stimulus, here for debugging purposes
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(stim)
        plt.xlabel("sample")
        plt.ylabel("amplitude (mA)")
        plt.title(title)
        plt.show()

    def quit_exp(self):
        self.win.close()
        core.quit()
        self.gvs.quit()


class Stimuli:

    def __init__(self, window, settings_dir, n_trials=0):
        """
        Create visual stimuli with PsychoPy.

        :param window: psychopy window instance
        :param settings_dir: directory where the stimulus settings are saved
        (stimuli.json)
        :param n_trials: (optional) number of trials for on pause screen
        """
        self.stimuli = OrderedDict()
        self.triggers = {}

        self.settings_dir = settings_dir
        self.num_trials = n_trials
        self.win = window

    def create(self):
        # read stimulus settings from json file
        stim_file = "{}/stimuli.json".format(self.settings_dir)
        with open(stim_file) as json_stim:
            stim_config = json.load(json_stim)

        # cycle through stimuli
        for key, value in stim_config.items():
            # get the correct stimulus class to call from the visual module
            stim_class = getattr(visual, value.get("stimType"))
            stim_settings = value.get("settings")
            self.stimuli[key] = stim_class(self.win, **stim_settings)
            # create stimulus trigger
            self.triggers[key] = False

        return self.stimuli, self.triggers

    def draw_pause_screen(self, current_trial):
        win_width, win_height = self.win.size
        pause_screen = visual.Rect(win=self.win, width=win_width,
                                   height=win_height, lineColor=(0, 0, 0),
                                   fillColor=(0, 0, 0))
        pause_str = "PAUSE  trial {}/{} Press space to continue".format(
            current_trial, self.num_trials)
        pause_text = visual.TextStim(win=self.win, text=pause_str,
                                     pos=(0.0, 0.0), color=(-1, -1, 0.6),
                                     units="pix", height=40)
        pause_screen.draw()
        pause_text.draw()
        self.win.flip()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    exp = WaveExp()
    exp.setup()
    exp.run()
    exp.quit_exp()
