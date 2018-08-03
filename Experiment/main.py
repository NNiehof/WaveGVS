import os
import logging
import multiprocessing
from queue import Empty
import json
import time
from collections import OrderedDict
import numpy as np
from psychopy import visual, core, event
from Experiment.GVSHandler import GVSHandler
from Experiment.loggingConfig import Listener, Worker
from Experiment.RandStim import RandStim

"""
Present sinusoidal GVS with a visual line oscillating around the
occipito-nasal axis at the same sine frequency. Participants can set the
amplitude of the visual line to match their subjective visual vertical, so
that they see the line as standing still and upright.
"""


class WaveExp:

    def __init__(self, sj=None, condition=""):

        self.debug = True

        # experiment settings and conditions
        self.sj = sj
        self.paradigm = "waveGVS"
        self.condition = condition
        self.f_sampling = 1e3
        self.screen_refresh_freq = 60
        self.duration_s = 14.0
        self.visual_soa = None
        self.current_mA = 1.0
        self.physical_channel_name = "cDAQ1Mod1/ao0"
        self.line_amplitude_step_size = 0.1
        self.phase_step_size = 0.5
        self.oled_delay = 0.05
        self.header = "trial_nr; current; frequency; line_offset; " \
                      "line_ori; amplitude; phase\n"
        self.phase = 0

        # longer practice trials
        if "practice" in self.condition:
            self.duration_s = 17.0
        # task: adapt phase as well as amplitude
        self.phase_shift = False
        if "phaseshift" in self.condition:
            self.phase_shift = True

        # initialise
        self.make_stim = None
        self.stimuli = None
        self.conditions = None
        self.trials = None
        self.triggers = None
        self.gvs_wave = None
        self.gvs_sent = None
        self.visual_time = None
        self.line_amplitude = 1.0
        self.line_angle = None
        self.visual_onset_delay = 0
        self.trial_nr = 0

        # root directory
        abs_path = os.path.abspath("__file__")
        self.root_dir = os.path.dirname(os.path.dirname(abs_path))
        self.settings_dir = "{}/Settings".format(self.root_dir)

    def setup(self):
        # display and window settings
        self._display_setup()

        # set up logging folder, file, and processes
        make_log = SaveData(self.sj, self.paradigm, self.condition,
                            file_type="log", sj_leading_zeros=3,
                            root_dir=self.root_dir)
        log_name = make_log.datafile
        self._logger_setup(log_name)
        main_worker = Worker(self.log_queue, self.log_formatter,
                             self.default_logging_level, "main")
        self.logger_main = main_worker.logger
        self.logger_main.debug("logger set up")

        # set up connection with galvanic stimulator
        self._gvs_setup()
        self._check_gvs_status("connected")

        # trial list
        if "practice" in self.condition:
            conditions_file = "{}/practice_conditions.json".format(
                self.settings_dir)
        else:
            conditions_file = "{}/conditions.json".format(self.settings_dir)
        with open(conditions_file) as json_file:
            self.conditions = json.load(json_file)
        self.trials = RandStim(**self.conditions)
        self.n_trials = self.trials.get_n_trials()

        # create stimuli
        self.make_stim = Stimuli(self.win, self.settings_dir, self.n_trials)
        self.stimuli, self.triggers = self.make_stim.create()

        # data save file
        self.save_data = SaveData(self.sj, self.paradigm, self.condition,
                                  sj_leading_zeros=3, root_dir=self.root_dir)
        self.save_data.write_header(self.header)

        self.logger_main.info("setup complete")

    def _display_setup(self):
        """
        Window and display settings
        """
        display_file = "{}/display.json".format(self.settings_dir)
        with open(display_file) as json_file:
            win_settings = json.load(json_file)
        self.win = visual.Window(**win_settings)
        self.mouse = event.Mouse(visible=False, win=self.win)

    def _logger_setup(self, log_file):
        """
        Establish a connection for parallel processes to log to a single file.

        :param log_file: str
        """
        # settings
        self.log_formatter = logging.Formatter("%(asctime)s %(processName)s %(thread)d %(message)s")
        if self.debug:
            self.default_logging_level = logging.DEBUG
        else:
            self.default_logging_level = logging.INFO

        # set up listener thread for central logging from all processes
        queue_manager = multiprocessing.Manager()
        self.log_queue = queue_manager.Queue()
        self.log_listener = Listener(self.log_queue, self.log_formatter,
                                     self.default_logging_level, log_file)
        # note: for debugging, comment out the next line. Starting the listener
        # will cause pipe breakage in case of a bug elsewhere in the code,
        # and the console will be flooded with error messages from the
        # listener.
        self.log_listener.start()

    def _gvs_setup(self):
        """
        Establish connection with galvanic stimulator
        """
        buffer_size = int(self.duration_s * self.f_sampling) + 1
        self.param_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        self.gvs_process = multiprocessing.Process(target=GVSHandler,
                                                   args=(self.param_queue,
                                                         self.status_queue,
                                                         self.log_queue,
                                                         buffer_size))
        self.gvs_process.start()

    def _check_gvs_status(self, key, from_queue=None, blocking=True):
        """
        Check the status of *key* on the status queue. Returns a boolean
        for the status. Note: this is a blocking process.
        :param key: str
        :param blocking: bool, set to True to hang until the key parameter
        is found in the queue. Set to False to check the queue once, then
        return.
        :return: bool or None
        """
        if from_queue is None:
            from_queue = self.status_queue
        while True:
            try:
                status = from_queue.get(block=blocking)
                if key in status:
                    return status[key]
            except Empty:
                return None
            if not blocking:
                return None

    def make_waves(self):
        """
        Make sine GVS signal and and antiphase sine visual line angle with
        the current trial's parameters.
        :return: gvs_wave, line_angle
        """
        self.gvs_time = np.arange(0, self.duration_s,
                             1.0 / self.f_sampling)
        gvs_wave = self.current_mA * np.sin(
            2 * np.pi * self.frequency * self.gvs_time)
        visual_duration = self.duration_s - (2 * self.visual_soa)
        visual_time = np.arange(0, visual_duration,
                                1.0 / self.screen_refresh_freq)
        # visual_wave = self.line_amplitude * -np.sin(
        #     2 * np.pi * self.frequency * visual_time)
        return gvs_wave, visual_time

    def next_line_orientation(self, t):
        """
        Calculate the next orientation of the visual line, with
        amplitude and phase changed through key presses.
        :param t: time sample
        :return: next_orientation
        """
        next_ori = self.line_amplitude * -np.sin(
            (2 * np.pi * self.frequency * t) - self.phase) + self.line_offset
        return next_ori

    def check_response(self):
        """
        Check for key presses, update the visual line amplitude
        """
        if self.phase_shift:
            key_response = event.getKeys(keyList=[
                "down", "up", "left", "right", "escape"])
        else:
            key_response = event.getKeys(keyList=["down", "up", "escape"])
        if key_response:
            if "down" in key_response:
                # only positive amplitudes
                if (self.line_amplitude - self.line_amplitude_step_size) >= 0:
                    self.line_amplitude -= self.line_amplitude_step_size
                else:
                    self.line_amplitude = 0
            elif "up" in key_response:
                self.line_amplitude += self.line_amplitude_step_size
            elif "left" in key_response:
                self.phase -= self.phase_step_size
            elif "right" in key_response:
                self.phase += self.phase_step_size
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

    def show_visual(self):
        """
        Visual loop that draws the stimuli on screen
        """
        self.triggers["rodStim"] = True
        line_start = time.time()

        for frame in self.visual_time:
            # line_angle = (frame * self.line_amplitude) + self.line_offset
            self.line_angle = self.next_line_orientation(frame)
            self.stimuli["rodStim"].setOri(self.line_angle)
            # save current line parameters in lists
            self.line_ori.append(self.line_angle)
            self.amplitudes.append(self.line_amplitude)
            self.phases.append(self.phase)
            # show stimulus on screen
            self.display_stimuli()
            self.frame_times.append(time.time())
            self.check_response()

        # log visual stimulus times
        line_end = time.time()
        self.logger_main.debug("{0} start visual stimulus".format(line_start))
        self.logger_main.debug("{0} stop visual stimulus".format(line_end))
        self.logger_main.info("visual stimulus duration = {0}".format(
            line_end - line_start))

        self.triggers["rodStim"] = False
        self.display_stimuli()

    def init_trial(self):
        """
        Initialise trial
        """
        self.logger_main.debug("initialising trial")
        trial = self.trials.get_stimulus(self.trial_nr)
        self.phase = 0
        # lists for saving the measured data
        self.line_ori = []
        self.amplitudes = []
        self.phases = []
        self.frame_times = []

        # trial parameters
        self.current_mA = trial[0]
        self.frequency = trial[1]
        self.line_offset = trial[2]
        self.line_amplitude = trial[3]

        # stimulus asynchrony: start visual one period after GVS
        self.visual_soa = 1.0 / self.frequency
        self.visual_onset_delay = self.visual_soa - self.oled_delay
        self.gvs_wave, self.visual_time = self.make_waves()
        # send GVS signal to handler
        self.param_queue.put(self.gvs_wave)
        self.logger_main.debug("wave sent to GVS handler")
        # check whether the gvs profile was successfully created
        if self._check_gvs_status("stim_created"):
            self.logger_main.info("gvs current profile created")
        else:
            self.logger_main.warning("WARNING: current profile not created")

    def wait_start(self):
        """
        Tell the participant to press the space bar to start the trial
        """
        self.triggers["startText"] = True
        # flush old key events before starting to listen
        event.clearEvents()
        while True:
            self.display_stimuli()
            start_key = event.getKeys(keyList=["space", "escape"])
            if "space" in start_key:
                self.triggers["startText"] = False
                self.display_stimuli()
                break
            elif "escape" in start_key:
                self.quit_exp()

    def _format_data(self):
        formatted_data = "{}; {}; {}; {}; {}; {}; {}\n".format(
            self.trial_nr, self.current_mA, self.frequency, self.line_offset,
            self.line_ori, self.amplitudes, self.phases)
        return formatted_data

    def run(self):
        """
        Run the experiment
        """
        for trial in range(self.n_trials):
            self.init_trial()
            self.trial_nr += 1

            # wait for space bar press to start trial
            self.wait_start()

            # send the GVS signal to the stimulator
            self.param_queue.put(True)

            # get onset time of GVS
            gvs_start = self._check_gvs_status("t_start_gvs")
            while True:
                if (time.time() - gvs_start) > self.visual_onset_delay:
                    break

            # clear old key events
            event.clearEvents()
            # draw visual line
            self.show_visual()

            # save data to file
            self.save_data.write(self._format_data())

            # get end time of GVS (blocks until GVS is finished)
            gvs_end = self._check_gvs_status("stim_sent")

            # self.stimulus_plot(self.visual_time, self.line_ori)
            # self.stimulus_plot(self.gvs_time, self.gvs_wave)
            # self.quit_exp()

        self.quit_exp()

    def stimulus_plot(self, xvals=None, stim=None, title=""):
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
        # send the stop signal to the GVS handler
        self.logger_main.info("quitting")
        self.param_queue.put("STOP")
        # wait for the GVS process to quit
        while True:
            if self._check_gvs_status("quit"):
                break
        # stop GVS and logging processes
        self.gvs_process.join()
        self.log_queue.put(None)
        self.log_listener.join()

        # close psychopy window and the program
        self.win.close()
        core.quit()


class SaveData:

    def __init__(self, sj, paradigm, condition, file_type="data",
                 sj_leading_zeros=0, root_dir=None):
        """
        Create a data folder and .txt or .log file, write data to file.

        :param sj: int, subject identification number
        :param paradigm: string
        :param condition: string
        :param file_type: type of file to create, either "data" (default)
        or "log" to make a log file.
        :param sj_leading_zeros: int (optional), add leading zeros to subject
        number until the length of sj_leading_zeros is reached.
        Example:
        with sj_leading_zeros=4, sj_name="2" -> sj_name="0002"
        :param root_dir: (optional) directory to place the Data folder in
        """
        # set up data folder
        if root_dir is None:
            abs_path = os.path.abspath("__file__")
            root_dir = os.path.dirname(os.path.dirname(abs_path))
        # set up subdirectory "Data" or "Log"
        assert(file_type in ["data", "log"])
        datafolder = "{}/{}".format(root_dir, file_type.capitalize())
        if not os.path.isdir(datafolder):
            os.mkdir(datafolder)

        # subject identifier with optional leading zeros
        sj_number = str(sj)
        if sj_leading_zeros > 0:
            while len(sj_number) < sj_leading_zeros:
                sj_number = "0{}".format(sj_number)

        # set up subject folder and data file
        subfolder = "{}/{}".format(datafolder, sj_number)
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        if file_type == "data":
            self.datafile = "{}/{}_{}_{}_{}.txt".format(subfolder, sj_number,
                                                        paradigm, condition,
                                                        timestr)
        else:
            self.datafile = "{}/{}_{}_{}_{}.log".format(subfolder, sj_number,
                                                        paradigm, condition,
                                                        timestr)

    def write_header(self, header):
        self.write(header)

    def write(self, data_str):
        with open(self.datafile, "a") as f:
            f.write(data_str)


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
    exp = WaveExp(sj=99, condition="")
    exp.setup()
    exp.run()
    exp.quit_exp()
