import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from IPython.display import Audio
from scipy.signal import spectrogram, butter, filtfilt


class geodata:

    def __init__(self, rx=0.0, ry=0.0, name="", delta_t=None, data=None):
        self.rx = rx
        self.ry = ry
        self.name = name
        self.delta_t = delta_t
        self.data = np.array([]) if data is None else np.asarray(data)

# Basic info methods


    def print(self):
        print(f"Name: {self.name}")
        print(f"Location: ({self.rx}, {self.ry})")
        print(f"Delta t: {self.delta_t}")
        print(f"Data length: {len(self.data)}")

    def distance_to(self, other):
        return np.sqrt((self.rx - other.rx)**2 + (self.ry - other.ry)**2)

    
    def plot(self):

        if self.delta_t is None:
            t = np.arange(len(self.data))
            plt.xlabel("Sample #")
        else:
            t = np.arange(len(self.data)) * self.delta_t
            plt.xlabel("Time (s)")
    
        plt.plot(t, self.data, label=self.name)
        plt.ylabel("Amplitude")

 # AUDIO METHODS 
   

    def load_m4a(self, filename):
        """
        Load an M4A file using librosa.
        sr=None keeps the original sampling rate.
        """
        audio, sampling_rate = librosa.load(filename, sr=None)
        self.data = audio
        self.delta_t = 1.0 / sampling_rate
        self.name = filename

    def play_audio(self):
        sampling_rate = int(1.0 / self.delta_t)
        return Audio(self.data, rate=sampling_rate)


# Frequency analysis
 

    def plot_spectrum(self, logscale=False):

        n = len(self.data)
        sampling_rate = 1.0 / self.delta_t

        X = np.fft.rfft(self.data)
        freqs = np.fft.rfftfreq(n, d=self.delta_t)
        amplitude = np.abs(X) / n

        plt.figure()
        if logscale:
            plt.semilogy(freqs, amplitude)
        else:
            plt.plot(freqs, amplitude)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum")
        plt.grid(True)
        plt.show()

    def plot_spectrogram(self, **kwargs):

        sampling_rate = 1.0 / self.delta_t
        f, t, Sxx = spectrogram(self.data, fs=sampling_rate, **kwargs)

        plt.figure()
        im = plt.pcolormesh(t, f, Sxx, shading="auto")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram")

        return im

# Low-pass filter
 

    def lpfilter(self, cutoff_hz, order=4):

        sampling_rate = 1.0 / self.delta_t
        nyq = sampling_rate / 2.0
        Wn = cutoff_hz / nyq

        b, a = butter(order, Wn, btype="low")
        self.data = filtfilt(b, a, self.data)


    def convert_phyphox_acceleration(filename):
        """
        Convert a Phyphox acceleration Excel file into
        a list of three geodata objects (x, y, z).
        """
    
        # Read Excel file (.xls or .xlsx)
        df = pd.read_excel(filename)
    
        # Compute delta_t from time column
        t = df["Time (s)"].values
        delta_t = float(np.median(np.diff(t)))
    
        # Extract acceleration components
        ax = df["Linear Acceleration x (m/s^2)"].values
        ay = df["Linear Acceleration y (m/s^2)"].values
        az = df["Linear Acceleration z (m/s^2)"].values
    
        # Create geodata objects (rx, ry undefined → use None)
        gx = geodata(rx=None, ry=None,
                     name="Linear Acceleration x",
                     delta_t=delta_t,
                     data=ax)
    
        gy = geodata(rx=None, ry=None,
                     name="Linear Acceleration y",
                     delta_t=delta_t,
                     data=ay)
    
        gz = geodata(rx=None, ry=None,
                     name="Linear Acceleration z",
                     delta_t=delta_t,
                     data=az)
    
        return [gx, gy, gz]