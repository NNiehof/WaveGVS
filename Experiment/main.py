import os
import logging
import multiprocessing
from queue import Empty
import json
import time
import threading
from collections import OrderedDict
import numpy as np
from psychopy import visual, core, event
from Experiment.GVSHandler import GVSHandler
from Experiment.arduino import ArduinoConnect, read_voltage
from Experiment.loggingConfig import Listener, Worker

"""
Present sinusoidal GVS with a visual line oscillating around the
occipito-nasal axis at the same sine frequency. Participants can set the
amplitude of the visual line to match their subjective visual vertical, so
that they see the line as standing still and upright.
"""


class WaveExp:

    def __init__(self, sj=None, condition=""):

        # experiment settings and conditions
        self.sj = sj
        self.paradigm = "waveGVS"
        self.condition = condition
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

        # set up logging folder, file, and processes
        make_log = SaveData(self.sj, self.paradigm, self.condition,
                            file_type="log", sj_leading_zeros=3,
                            root_dir=self.root_dir)
        log_name = make_log.datafile
        self._logger_setup(log_name)
        main_worker = Worker(self.log_queue, self.log_formatter,
                             self.default_logging_level, "main")
        self.logger_main = main_worker.logger

        # set up connection with galvanic stimulator
        self._gvs_setup()
        self._check_gvs_status("connected")

        # connect to Arduino for retrieving GVS signals sent to the stimulator
        self.gvs_sent = ArduinoConnect(device_name="Arduino", baudrate=9600)
        self.gvs_sent.connect()

        # create stimuli
        self.make_stim = Stimuli(self.win, self.settings_dir, self.n_trials)
        self.stimuli, self.triggers = self.make_stim.create()

        # data save file
        self.save_data = SaveData(self.sj, self.paradigm, self.condition,
                                  sj_leading_zeros=3, root_dir=self.root_dir)

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
        self.default_logging_level = logging.DEBUG

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

    def _check_gvs_status(self, key, blocking=True):
        """
        Check the status of *key* on the status queue. Returns a boolean
        for the status. Note: this is a blocking process.
        :param key: str
        :param blocking: bool, set to True to hang until the key parameter
        is found in the queue. Set to False to check the queue once, then
        return.
        :return: bool or None
        """
        while True:
            try:
                status = self.status_queue.get(block=blocking)
                if key in status:
                    return status[key]
            except Empty:
                return None
            if not blocking:
                return None

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
        self.gvs_wave, self.visual_wave = self.make_waves(
            frequency, self.line_amplitude)
        # send GVS signal to handler
        self.param_queue.put(self.gvs_wave)
        # check whether the gvs profile was successfully created
        if self._check_gvs_status("stim_created"):
            self.logger_main.info("gvs current profile created")
        else:
            self.logger_main.warning("WARNING: current profile not created")

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
            start_key = event.getKeys(keyList=["space", "escape"])
            if "space" in start_key:
                self.triggers["startText"] = False
                self.display_stimuli()
                break
            elif "escape" in start_key:
                self.quit_exp()

    def run(self):
        """
        Run the experiment
        """
        # start reading from Arduino
        self.ard_reader = threading.Thread(target=read_voltage,
                                           args=(self.gvs_sent.serial_in,))
        self.ard_reader.start()
        for trial in self.trial_list:

            self.init_trial(trial)

            # wait for space bar press to start trial
            self.wait_start()

            # send the GVS signal to the stimulator
            self.param_queue.put(True)

            # draw visual line
            self.show_visual()
            # self.stimulus_plot(self.line_ori, self.frame_times)
        self.quit_exp()

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
        self.ard_reader.join()


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
    import matplotlib.pyplot as plt
    exp = WaveExp(sj=99, condition="")
    exp.setup()
    exp.run()
    exp.quit_exp()
