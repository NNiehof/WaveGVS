# Nynke Niehof, 2018

import numpy as np
from Experiment.GVS import GVS
from Experiment.loggingConfig import Worker, formatter, default_logging_level


class GVSHandler():
    def __init__(self, param_queue, status_queue, logging_queue, buffer_size):
        PHYSICAL_CHANNEL_NAME = ["cDAQ1Mod1/ao0", "cDAQ1Mod1/ao1"]
        SAMPLING_FREQ = 1e3

        # I/O queues
        self.param_queue = param_queue
        self.status_queue = status_queue
        self.logging_queue = logging_queue
        self.stimulus = []

        # set up logger
        worker = Worker(logging_queue, formatter, default_logging_level,
                        "GVSHandler")
        self.logger = worker.logger
        # second logger to pass to GVS object
        subworker = Worker(logging_queue, formatter, default_logging_level,
                           "GVS")
        self.sublogger = subworker.logger

        # GVS control object
        self.gvs = GVS(logger=self.sublogger)
        self.buffer_size = int(buffer_size)
        timing = {"rate": SAMPLING_FREQ, "samps_per_chan": self.buffer_size}
        connected = self.gvs.connect(PHYSICAL_CHANNEL_NAME, **timing)
        if connected:
            self.logger.info("NIDAQ connection established")
            self.status_queue.put({"connected": True})
        else:
            self.logger.info("NIDAQ connection failed")
            self.status_queue.put({"connected": False})

        # GVSHandler can't be a subclass of multiprocessing.Process, as the
        # GVS object contains ctypes pointers and can't be pickled.
        # GVSHandler's methods can't be accessed from the parent process.
        # As a workaround, the event loop is started by calling the run method
        # here at the end of the initialisation.
        self.run()

    def run(self):
        """
        Event loop that listens for queue input. Input of type *dict* is used
        for stimulus creation, input of type *int* is used to trigger onset of
        GVS stimulation. Input "STOP" to exit the method.
        This event loop is automatically started after a GVSHandler object
        is initialised.
        """
        while True:
            data = self.param_queue.get()
            if isinstance(data, str) & data == "STOP":
                quit_gvs = self.gvs.quit()
                if quit_gvs:
                    self.status_queue.put({"quit": True})
                else:
                    self.status_queue.put({"quit": False})
                break

            else:
                if isinstance(data, np.ndarray):
                    self.stimulus = self._analog_feedback_loop(data)

                elif isinstance(data, bool) & (data is True):
                    self._send_stimulus()

                else:
                    self.logger.error("Incorrect input to GVSHandler parameter"
                                      " queue. Input must be a dict with "
                                      "parameters specified in GVS.py, a "
                                      "boolean, or a string STOP to quit.")
                    self.status_queue.put({"stim_created": False})

    def _analog_feedback_loop(self, gvs_wave, start_end_blip_voltage=2.5):
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
        duplicate_wave[-1] = -start_end_blip_voltage

        # add voltage reset (0 sample) at the end
        gvs_wave = np.append(gvs_wave, 0)
        duplicate_wave = np.append(duplicate_wave, 0)
        stimulus = np.stack((duplicate_wave, gvs_wave), axis=0)
        self.status_queue.put({"stim_created": True})
        return stimulus

    def _send_stimulus(self):
        """
        Send the stimulus to the GVS channel, check whether all samples
        were successfully written
        """
        n_samples = None
        samps_written = 0
        try:
            samps_written = self.gvs.write_to_channel(self.stimulus,
                                                      reset_to_zero_volts=False)
            n_samples = len(self.stimulus)
            # delete stimulus after sending, so that it can only be sent once
            self.stimulus = None
        except AttributeError as err:
            self.logger.error("Error: tried to send invalid stimulus to NIDAQ."
                              "\nNote that a stimulus instance can only be"
                              " sent once.\nAttributeError: {}".format(err))
        self.logger.info("GVS: {} samples written".format(samps_written))

        if n_samples == samps_written:
            self.status_queue.put({"stim_sent": True})
        else:
            self.status_queue.put({"stim_sent": False})
