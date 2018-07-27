# Nynke Niehof, 2018

import time
from queue import Empty
from Experiment.loggingConfig import Worker, formatter, default_logging_level
from Experiment.arduino import ArduinoConnect


class ArduinoHandler:

    def __init__(self, in_queue, return_queue, logging_queue,
                 device_name="Arduino", baudrate=9600):
        # I/O queues
        self.in_queue = in_queue
        self.return_queue = return_queue
        self.logging_queue = logging_queue

        # timestamp for signal start and end
        self.trigger_time = None

        # set up logger
        worker = Worker(logging_queue, formatter, default_logging_level,
                        "ArduinoHandler")
        self.logger = worker.logger
        # second logger to pass to ArduinoConnect object
        subworker = Worker(logging_queue, formatter, default_logging_level,
                           "ArduinoConnect")
        self.sublogger = subworker.logger

        # connect to Arduino
        self.arduino = ArduinoConnect(device_name=device_name,
                                      baudrate=baudrate,
                                      logger=self.sublogger)
        connected = self.arduino.connect()
        if connected:
            self.return_queue.put({"connected": True})
        else:
            self.return_queue.put({"connected": False})

        # start serial readout loop
        self.run()

    def run(self):
        get_trigger = True

        while True:
            signal_in = None
            timestamp = None
            number_read = None
            try:
                signal_in = self.in_queue.get(block=False)
                if signal_in == "STOP":
                    self.arduino.quit()
                    break
            except Empty:
                # do nothing if there is no input
                pass

            if get_trigger:
                # get time of first or last sample
                measurement = self.arduino.read_voltage()
                if measurement:
                    number_read = measurement[0]
                    timestamp = measurement[1]
                if timestamp is not None and (number_read in [634, 635]):
                    self.trigger_time = timestamp
                    get_trigger = False
                    self.return_queue.put(self.trigger_time)
            elif self.trigger_time and ((time.time() - self.trigger_time) > 0.1):
                # after 0.1 s, start looking for the next trigger sample
                get_trigger = True
                self.trigger_time = None
