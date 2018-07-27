"""
ArduinoConnect detects an Arduino device connected to a serial port,
establishes a connection, and returns the port object.

Arduino Uno and similar devices can commonly be found as "ttyACM0" on Linux and
as "Arduino" on Windows.
"""

import serial
import serial.tools.list_ports
import time
import logging


class ArduinoConnect:

    def __init__(self, device_name="ttyACM0", baudrate=9600, logger=None):
        self.device_name = device_name
        self.baudrate = baudrate
        self.cport = None
        self.serial_in = None
        self.voltage = 0

        # set up logger
        if logger is not None:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.DEBUG,
                                format="%(asctime)s %(message)s")
            self.logger = logging.getLogger()

    def connect(self):
        """
        Finds which COM-port has an Arduino and returns the port object. If
        no Arduino is found, returns None.
        """
        comports = list(serial.tools.list_ports.comports())
        for port in comports:
            if self.device_name in port[1]:
                self.cport = port[0]
        if not self.cport:
            self.logger.warning("Warning: no Arduino detected")
            return self.cport

        # Connect to Arduino, wait for it to initialise
        self.serial_in = serial.Serial(self.cport, self.baudrate, timeout=5)
        # waits until Arduino sends a serial signal, or until timeout
        signal_received = self.serial_in.read()

        if signal_received:
            self.logger.info("Connection with Arduino established")
            return True
        else:
            self.logger.warning("Warning: no signal from Arduino detected, check connection")
            return False

    def read_voltage(self):
        """
        Arduino send unsigned integer from 0 - 1023 which
        must be translated to a voltage between -5 V and 5 V.
        :return: voltage
        """
        digital_in = self.serial_in.readline()
        start_time = time.time()
        number_in = None
        digital_in = digital_in.decode()
        if digital_in:
            number_in = int(digital_in.rstrip())
            return number_in, start_time
    # return 2.0 * (digital_in * (5.0 / 1023.0)) - 5.0

    def quit(self):
        """
        Close the serial port connection.
        """
        self.serial_in.close()
