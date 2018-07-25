import os
import json
import time
from collections import OrderedDict
import numpy as np
from psychopy import visual, core, event
from Experiment.GVS import GVS
from Experiment.arduino import ArduinoConnect

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
        self.physical_channel_name = ["cDAQ1Mod1/ao0", "cDAQ1Mod1/ao1"]
        self.line_amplitude_step_size = 0.5

        # TODO: generate trial list
        # frequency, amplitude
        self.trial_list = [[2.0, 1.0], [1.0, 1.5], [0.2, 3.0],
                           [2.0, 3.0], [1.0, 1.0], [0.2, 1.5],
                           [2.0, 1.5], [1.0, 3.0], [0.2, 1.0]]

        # initialise
        self.make_stim = None
        self.stimuli = None
        self.triggers = None
        self.gvs_wave = None
        self.gvs_sent = None
        self.visual_wave = None
        self.line_amplitude = 1.0
        self.stop_trial = False

        # root directory
        abs_path = os.path.abspath("__file__")
        self.root_dir = os.path.dirname(os.path.dirname(abs_path))
        self.settings_dir = "{}/Settings".format(self.root_dir)

    def setup(self):
        # display and window settings
        self._display_setup()

        # set up connection with galvanic stimulator
        self._gvs_setup()

        # connect to Arduino for retrieving GVS signals sent to the stimulator
        self.gvs_sent = ArduinoConnect(device_name="Arduino", baudrate=9600)
        self.gvs_sent.connect()

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
        buffer_size = int(self.duration_s * self.f_sampling) + 1
        timing = {"rate": self.f_sampling, "samps_per_chan": buffer_size}
        self.gvs = GVS()
        self.is_connected = self.gvs.connect(self.physical_channel_name,
                                             **timing)

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
                                1.0 / self.screen_refresh_freq)
        visual_wave = line_amplitude * -np.sin(2 * np.pi * frequency * visual_time)
        return gvs_wave, visual_wave

    def _analog_feedback_loop(self, gvs_wave, start_end_blip_voltage):
        """
        Add a copy of the GVS signal to send to a second channel via the NIDAQ.
        The copy (but not the GVS signal) has a 2.5 V blip of a single sample
        at the start and the end, to signal the onset and end of the
        stimulation. In the signal that is sent to the GVS channel (here:
        channel A0), the first and last sample are zero.
        Also, an extra zero sample is added to the end of both signals,
        to reset the voltage to baseline.

        :param gvs_wave: GVS signal
        :param start_end_blip_voltage: voltage to give to first and last
        as a signal. Voltage should not be present in the rest of the waveform.
        :return: stacked signal, with second row being the original GVS signal,
        the first row being the copied signal with first and last sample
        changed to 2.5 V.
        """
        duplicate_wave = gvs_wave[:]
        # blip at start and end of copied GVS wave
        duplicate_wave[0] = start_end_blip_voltage
        duplicate_wave[-1] = start_end_blip_voltage

        # add voltage reset (0 sample) at the end
        gvs_wave = np.append(gvs_wave, 0)
        duplicate_wave = np.append(duplicate_wave, 0)
        return np.stack((duplicate_wave, gvs_wave), axis=0)

    def check_response(self):
        """
        Check for key presses, update the visual line amplitude
        """
        key_response = event.getKeys(keyList=["left", "right", "return", "escape"])
        if key_response:
            if "left" in key_response:
                self.line_amplitude -= self.line_amplitude_step_size
            elif "right" in key_response:
                self.line_amplitude += self.line_amplitude_step_size
            elif "return" in key_response:
                self.stop_trial = True
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

    def init_trial(self, trial):
        """
        Initialise trial
        """
        self.line_ori = []
        self.frame_times = []
        self.stop_trial = False
        frequency = trial[0]
        self.line_amplitude = trial[1]
        gvs_wave, self.visual_wave = self.make_waves(
            frequency, self.line_amplitude)
        self.gvs_wave = self._analog_feedback_loop(gvs_wave, 2.5)

    def show_visual(self):
        """
        Visual loop that draws the stimuli on screen
        """
        self.triggers["rodStim"] = True
        line_start = time.time()

        for frame in self.visual_wave:
            self.stimuli["rodStim"].setOri(frame * self.line_amplitude)
            self.line_ori.append(frame * self.line_amplitude)
            self.display_stimuli()
            self.frame_times.append(time.time())
            self.check_response()
            if self.stop_trial:
                resp_time = time.time() - line_start
                print("response time: {} s".format(resp_time))
                break

        self.triggers["rodStim"] = False
        self.display_stimuli()

    def wait_start(self):
        """
        Tell the participant to press the space bar to start the trial
        """
        self.triggers["startText"] = True
        while True:
            self.display_stimuli()
            start_key = event.getKeys("space")
            if "space" in start_key:
                self.triggers["startText"] = False
                self.display_stimuli()
                break

    def run(self):
        """
        Run the experiment
        """
        for trial in self.trial_list:

            self.init_trial(trial)

            # wait for space bar press to start trial
            self.wait_start()

            # send GVS signal
            if self.is_connected:
                self.gvs.write_to_channel(self.gvs_wave,
                                          reset_to_zero_volts=False)
            self.gvs_sent.read_voltage()

            # draw visual line
            self.show_visual()
            # self.stimulus_plot(self.line_ori, self.frame_times)
            # self.quit_exp()

    def stimulus_plot(self, stim, xvals=None, title=""):
        """
        Plot generated stimulus, here for debugging purposes
        """
        import matplotlib.pyplot as plt
        plt.figure()
        if xvals is not None:
            plt.plot(xvals, stim)
        else:
            plt.plot(stim)
        plt.xlabel("time")
        plt.ylabel("amplitude")
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
