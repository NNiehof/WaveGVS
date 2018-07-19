"""
ArduinoConnect detects an Arduino device connected to a serial port,
establishes a connection, and returns the port object.

Arduino Uno and similar devices can commonly be found as "ttyACM0" on Linux and
as "Arduino" on Windows.
"""

import serial
import serial.tools.list_ports


class ArduinoConnect:

    def __init__(self, device_name="ttyACM0", baudrate=9600):
        self.device_name = device_name
        self.baudrate = baudrate
        self.cport = None
        self.serial_in = None

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
            print("Warning: no Arduino detected")
            return self.cport

        # Connect to Arduino, wait for it to initialise
        self.serial_in = serial.Serial(self.cport, self.baudrate, timeout=2)
        # waits until Arduino sends a serial signal, or until timeout
        signal_received = self.serial_in.read()

        if signal_received:
            print("Connection with Arduino established")
        else:
            print("Warning: no signal from Arduino detected, check connection")
        return self.serial_in
