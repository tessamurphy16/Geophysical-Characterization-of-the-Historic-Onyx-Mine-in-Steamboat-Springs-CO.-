import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
from IPython.display import Audio
import plotly.express as px
from scipy.signal import spectrogram, butter, filtfilt, correlate
import segyio

class geodata:

    def __init__(self, rx=0.0, ry=0.0, rz=0.0, sx=0.0, sy=0.0, sz=0.0,
                 offset=0.0, name="", delta_t=None, data=None):
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.offset = offset
        self.name = name
        self.delta_t = delta_t
        self.data = np.array([]) if data is None else np.asarray(data)

    def copy(self):

        return geodata(
            rx=self.rx,
            ry=self.ry,
            rz=self.rz,
            sx=self.sx,
            sy=self.sy,
            sz=self.sz,
            offset=self.offset,
            name=self.name,
            delta_t=self.delta_t,
            data=np.copy(self.data)
        )


    def plot_vertical_trace(self, scale=1.0, offset=0.0):
        if self.data.size == 0:
            return
    
        time = np.arange(len(self.data)) * self.delta_t
    
        tr = np.asarray(self.data).copy()
        trmax = np.max(np.abs(tr))
        if trmax == 0:
            trmax = 1.0
    
        tr_plot = tr / trmax * scale + offset
    
        plt.plot(tr_plot, time, label=self.name, c='black', linewidth=1)
        plt.fill_betweenx(time, offset, tr_plot, where=(tr_plot > offset), color='black', alpha=0.5)
        plt.ylabel("time (ms)")

    
# Basic info methods


    def print(self):
        print(f"Name: {self.name}")
        print(f"Location: ({self.rx}, {self.ry})")
        print(f"Delta t: {self.delta_t}")
        print(f"Data length: {len(self.data)}")

    def distance_to(self, other):
        return np.sqrt((self.rx - other.rx)**2 + (self.ry - other.ry)**2)

    def cal_dist(self, sx, sz):
        return ((self.x - sx)**2 + (self.z - sz)**2)**0.5
    
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

    def get_taxis(self):
        taxis = np.arange(len(self.data)) * self.delta_t
        return taxis

    def plot_interactive(self):
        df = pd.DataFrame()
        df['time'] = self.get_taxis()
        df['v'] = self.data

        fig = px.line(data_frame=df, x='time', y='v')
        fig.show()



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

class SeismicGather:

    def __init__(self):
        self.data = []

    def read_synthetic_data(self, filename):
        syn = np.load(filename)
    
        rx = syn['rx']
        rz = syn['rz']
        sx = float(syn['sx'])
        sz = float(syn['sz'])
        data = syn['data']
        self.dt = float(syn['dt'])
        dt = float(syn['dt'])
    
        self.sx = sx
        self.sz = sz
        self.data = []
    
        nrec = data.shape[1]
    
        for i in range(nrec):
            trace = data[:, i]
    
            g = geodata(
                rx=float(rx[i]),
                rz=float(rz[i]),
                sx=sx,
                sz=sz,
                name=f"rec_{i}",
                delta_t=dt,
                data=trace
            )
    
            # signed horizontal offset for CMP/NMO
            g.offset = float(rx[i] - sx)
    
            self.data.append(g)

    def wiggle_plot(self, xaxis='rx', scale=1.0):
        plt.figure(figsize=(8, 7))
    
        for trace in self.data:
            pos = getattr(trace, xaxis) if hasattr(trace, xaxis) else trace.offset
            trace.plot_vertical_trace(scale=scale, offset=pos)
    
        plt.gca().invert_yaxis()
        

    def remove_source_wavelet(self, src_wavelet):

        # FFT of source wavelet
        src_wavelet = np.asarray(src_wavelet)
        eps = 1e-10
    
        # loop through each trace object in the gather
        for trace_obj in self.data:
    
            xcor = correlate(trace_obj.data, src_wavelet, mode='full')
            trace_obj.data = xcor[xcor.size//2:]

    def bpfilter(self, fmin, fmax):
        """
        Apply a bandpass filter to all traces in the seismic gather.
    
        Parameters
        ----------
        fmin : float
            minimum frequency (Hz)
        fmax : float
            maximum frequency (Hz)
        """
    
        for trace_obj in self.data:
    
            trace = np.asarray(trace_obj.data)
    
            dt = trace_obj.delta_t
    
            fs = 1.0 / dt
            nyq = fs / 2
    
            low = fmin / nyq
            high = fmax / nyq
    
            b, a = butter(4, [low, high], btype='band')
    
            trace_filtered = filtfilt(b, a, trace)
    
            trace_obj.data = trace_filtered
   
    def get_taxis(self):
        if len(self.data) == 0:
            return np.array([])
        return np.arange(len(self.data[0].data)) * self.data[0].delta_t

    def get_offsets(self):
        if len(self.data) == 0:
            return np.array([])
        return np.array([tr.offset for tr in self.data], dtype=float)
        
    def select_offset_range(self, min_offset, max_offset):
        gnew = SeismicGather()
        gnew.data = []

        for tr in self.data:
            if min_offset <= abs(tr.offset) <= max_offset:
                gnew.data.append(tr.copy())
    
        return gnew.sort_by_offset()

    def get_midpoints(self):
        if len(self.data) == 0:
            return np.array([])
        return np.array([(tr.sx + tr.rx) / 2.0 for tr in self.data], dtype=float)

    def get_data_matrix(self):
        if len(self.data) == 0:
            return np.array([[]])
        return np.column_stack([tr.data for tr in self.data])

    def sort_by_offset(self):
        if len(self.data) == 0:
            return self

        idx = np.argsort(self.get_offsets())
        gnew = SeismicGather()
        gnew.data = [self.data[i].copy() for i in idx]

        if hasattr(self, 'sx'):
            gnew.sx = self.sx
        if hasattr(self, 'sz'):
            gnew.sz = self.sz

        return gnew

    def midpoint_histogram(self, bins=50):
        mids = self.get_midpoints()

        plt.figure(figsize=(9, 4))
        plt.hist(mids, bins=bins, edgecolor='k')
        plt.xlabel("Midpoint x")
        plt.ylabel("Trace count")
        plt.title("Midpoint Density Histogram")
        plt.grid(alpha=0.3)
        plt.show()

    def select_cmp_gather(self, Mx, half_bin=50.0):
        mids = self.get_midpoints()
        mask = (mids > Mx - half_bin) & (mids <= Mx + half_bin)

        gnew = SeismicGather()
        gnew.data = [tr.copy() for tr, keep in zip(self.data, mask) if keep]

        return gnew.sort_by_offset()

    def make_one_sided_cmp(self, decimals=3):
        if len(self.data) == 0:
            return SeismicGather()

        offsets = np.abs(self.get_offsets())
        offsets_round = np.round(offsets, decimals)
        unique_offsets = np.unique(offsets_round)

        data2d = self.get_data_matrix()
        dt = self.data[0].delta_t

        gnew = SeismicGather()
        gnew.data = []

        for off in unique_offsets:
            mask = offsets_round == off
            avg_trace = np.mean(data2d[:, mask], axis=1)

            tr = geodata(
                rx=float(off),
                sx=0.0,
                delta_t=dt,
                data=avg_trace,
                name=f"off_{off}"
            )
            tr.offset = float(off)
            gnew.data.append(tr)

        return gnew.sort_by_offset()

    def tnmo_shift_ms(self, offset_m, t0_ms, vrms_mps):
        t0_s = t0_ms / 1000.0
        tnmo_s = np.sqrt(t0_s**2 + (offset_m**2) / (vrms_mps**2))
        shift_s = tnmo_s - t0_s
        return shift_s * 1000.0

    def shift_gather_for_event(self, t0_ms, vrms_mps):
        taxis = self.get_taxis()
        offsets = self.get_offsets()

        gnew = SeismicGather()
        gnew.data = []

        for i, tr in enumerate(self.data):
            shift_ms = self.tnmo_shift_ms(offsets[i], t0_ms, vrms_mps)

            shifted_trace = np.interp(
                taxis + shift_ms,
                taxis,
                tr.data,
                left=0.0,
                right=0.0
            )

            tr_new = tr.copy()
            tr_new.data = shifted_trace
            gnew.data.append(tr_new)

        return gnew

    def semblance_one_event(self, t0_ms, vrms_mps, window_ms=200):
        shifted = self.shift_gather_for_event(t0_ms, vrms_mps)

        taxis = shifted.get_taxis()
        data2d = shifted.get_data_matrix()

        half_w = window_ms / 2.0
        t_ind = (taxis >= t0_ms - half_w) & (taxis <= t0_ms + half_w)

        if np.sum(t_ind) == 0:
            return np.nan

        data_win = data2d[t_ind, :]
        N = data_win.shape[1]

        if N < 2:
            return np.nan

        S = np.sum(np.sum(data_win, axis=1)**2)
        E = np.sum(data_win**2)

        if E == 0:
            return np.nan

        R = (S - E) / ((N - 1) * E)
        return R

    def compute_semblance_map(self, t0_values, vrms_values, window_ms=200):
        semblance_map = np.full((len(t0_values), len(vrms_values)), np.nan)

        for j, t0_val in enumerate(t0_values):
            for i, vrms_val in enumerate(vrms_values):
                semblance_map[j, i] = self.semblance_one_event(
                    t0_val,
                    vrms_val,
                    window_ms=window_ms
                )

        return semblance_map

    def build_vrms_profile(self, t0_picks, vrms_picks):
        taxis = self.get_taxis()

        vrms_profile = np.interp(
            taxis,
            np.asarray(t0_picks, dtype=float),
            np.asarray(vrms_picks, dtype=float),
            left=float(vrms_picks[0]),
            right=float(vrms_picks[-1])
        )

        return vrms_profile

    def nmo_full_profile(self, vrms_profile):
        taxis = self.get_taxis()
        t0_s = taxis / 1000.0

        gnew = SeismicGather()
        gnew.data = []

        for i, tr in enumerate(self.data):
            offset = tr.offset

            t_orig_s = np.sqrt(t0_s**2 + (offset**2) / (vrms_profile**2))
            t_orig_ms = t_orig_s * 1000.0

            corrected = np.interp(
                t_orig_ms,
                taxis,
                tr.data,
                left=0.0,
                right=0.0
            )

            tr_new = tr.copy()
            tr_new.data = corrected
            gnew.data.append(tr_new)

        return gnew

    def mute_nmo_gather(self, t0_at_zero=300.0, t_at_2000=1300.0, offset_ref=2000.0):
        taxis = self.get_taxis()
        offsets = np.abs(self.get_offsets())

        mute_times = t0_at_zero + (offsets / offset_ref) * (t_at_2000 - t0_at_zero)

        gnew = SeismicGather()
        gnew.data = []

        for tr, tmute in zip(self.data, mute_times):
            tr_new = tr.copy()
            new_data = np.asarray(tr.data, dtype=float).copy()
            new_data[taxis < tmute] = np.nan
            tr_new.data = new_data
            gnew.data.append(tr_new)

        return gnew, mute_times

    def stack_nanmean(self, name="stack_trace"):
        data2d = np.asarray(self.get_data_matrix(), dtype=float)

        valid_counts = np.sum(~np.isnan(data2d), axis=1)
        stack_sum = np.nansum(data2d, axis=1)

        stack = np.divide(
            stack_sum,
            valid_counts,
            out=np.zeros_like(stack_sum),
            where=valid_counts > 0
        )

        g = geodata(
            name=name,
            delta_t=self.data[0].delta_t,
            data=stack
        )
        return g

    def get_midpoints_offsets(self):
        """
        Return midpoint x positions, offsets, and trace amplitudes
        for all traces in this gather/survey.
        Assumes each trace has attributes:
            tr.sx, tr.rx, tr.data, tr.dt
        """
        midpoints = []
        offsets = []
        traces = []

        for tr in self.data:
            sx = float(tr.sx)
            rx = float(tr.rx)
            midpoints.append(0.5 * (sx + rx))
            offsets.append(rx - sx)
            traces.append(np.asarray(tr.data).copy())

        return np.array(midpoints), np.array(offsets), np.array(traces)

    def get_time_axis(self):
        """
        Return time axis in seconds using the first trace.
        """
        nt = len(self.data[0].data)
        dt = float(self.dt) / 1000
        return np.arange(nt) * dt

    def get_fold_bins(self, bin_size=100.0):
        """
        Bin traces by midpoint and return bin centers + fold counts.
        """
        midpoints, _, _ = self.get_midpoints_offsets()

        xmin = np.min(midpoints)
        xmax = np.max(midpoints)

        edges = np.arange(xmin, xmax + bin_size, bin_size)
        counts, edges = np.histogram(midpoints, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

        return centers, counts, edges

    def extract_cdp_gather(self, cdp_x, bin_size=100.0):
        """
        Extract traces whose midpoint falls inside one CDP bin.
        Returns:
            data_mat : (nt, ntraces)
            offsets  : (ntraces,)
            t        : (nt,)
        """
        midpoints, offsets, traces = self.get_midpoints_offsets()
        half = bin_size / 2.0

        mask = (midpoints >= cdp_x - half) & (midpoints < cdp_x + half)

        if np.sum(mask) == 0:
            return None, None, None

        offsets_sel = offsets[mask]
        traces_sel = traces[mask]

        # sort by offset
        order = np.argsort(offsets_sel)
        offsets_sel = offsets_sel[order]
        traces_sel = traces_sel[order]

        # traces_sel shape: (ntr, nt) -> transpose to (nt, ntr)
        data_mat = traces_sel.T
        t = self.get_time_axis()

        return data_mat, offsets_sel, t

    def nmo_correct_matrix(self, data_mat, offsets, t, vrms):
        """
        Apply NMO correction using constant Vrms.
        data_mat: (nt, ntr)
        offsets: (ntr,)
        t: time axis in seconds
        vrms: m/s
        """
        nt, ntr = data_mat.shape
        out = np.zeros_like(data_mat)
    
        for i in range(ntr):
            x = offsets[i]
    
            for it0, t0 in enumerate(t):
                tn = np.sqrt(t0**2 + (x / vrms)**2)
    
                if tn <= t[-1]:
                    out[it0, i] = np.interp(tn, t, data_mat[:, i], left=0.0, right=0.0)
    
        return out

    def mute_large_offsets(self, data_mat, offsets, max_offset=None):
        """
        Zero traces whose absolute offset exceeds max_offset.
        """
        out = data_mat.copy()
        if max_offset is None:
            return out

        mask = np.abs(offsets) > max_offset
        out[:, mask] = 0.0
        return out

    def stack_cdp(self, data_mat):
        """
        Stack along trace axis.
        """
        return np.mean(data_mat, axis=1)

    def build_stacked_section(self, vrms, bin_size=100.0, fold_frac=0.8, max_offset=None):
        """
        Build 2D stacked time section for all high-fold CDPs.
        Returns:
            section : (nt, ncdp)
            t       : (nt,)
            cdp_xs  : (ncdp,)
            folds   : (ncdp,)
        """
        centers, counts, _ = self.get_fold_bins(bin_size=bin_size)
        max_fold = np.max(counts)

        keep = counts >= fold_frac * max_fold
        cdp_xs = centers[keep]
        folds = counts[keep]

        stack_list = []
        t_ref = None

        for cdp_x in cdp_xs:
            data_mat, offsets, t = self.extract_cdp_gather(cdp_x, bin_size=bin_size)

            if data_mat is None:
                continue

            nmo = self.nmo_correct_matrix(data_mat, offsets, t, vrms)
            nmo_muted = self.mute_large_offsets(nmo, offsets, max_offset=max_offset)
            stk = self.stack_cdp(nmo_muted)

            stack_list.append(stk)
            t_ref = t

        section = np.column_stack(stack_list)
        return section, t_ref, cdp_xs, folds

    def time_to_depth_section(self, section_t, t, vrms):
        """
        Convert stacked section from two-way time to depth.
        z = vrms * t / 2
        """
        z = vrms * t / 2.0
        return section_t, z

    def plot_cdp_gather(self, data_mat, offsets, t, title="CDP Gather", scale=0.75, gain_power=0.15):
        plt.figure(figsize=(6, 8))
    
        if len(offsets) > 1:
            dx = np.median(np.diff(offsets))
        else:
            dx = 100.0
    
        wiggle_scale = scale * dx
    
        gain = np.ones_like(t)
        if gain_power > 0:
            gain = 1.0 + (t / np.max(t)) ** gain_power
    
        for i in range(data_mat.shape[1]):
            tr = data_mat[:, i].copy() * gain
    
            # use a panel-based scaling, not full per-trace scaling
            amax = np.percentile(np.abs(data_mat), 98)
            if amax == 0:
                amax = 1.0
    
            tr = tr / amax
            plt.plot(offsets[i] + tr * wiggle_scale, t * 1000, 'k', linewidth=1.1)
    
        plt.gca().invert_yaxis()
        plt.xlabel("Offset (m)")
        plt.ylabel("Time (ms)")
        plt.title(title)
        plt.xlim(-50, np.max(offsets) + dx)
        plt.grid(True, alpha=0.25)
        plt.show()
    
    def plot_section(self, section, yaxis, xaxis, ylabel="Time (s)", title="Stacked Section"):
        plt.figure(figsize=(10, 6))
        vmax = np.percentile(np.abs(section), 99)

        plt.imshow(
            section,
            cmap="gray",
            aspect="auto",
            extent=[xaxis[0], xaxis[-1], yaxis[-1], yaxis[0]],
            vmin=-vmax,
            vmax=vmax
        )
        plt.xlabel("CDP X (m)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def extract_cdp_gather_display(self, cdp_x, bin_size=200.0, positive_only=True, step=1):
        midpoints, offsets, traces = self.get_midpoints_offsets()
        half = bin_size / 2.0
    
        mask = (midpoints >= cdp_x - half) & (midpoints < cdp_x + half)
    
        if np.sum(mask) == 0:
            return None, None, None
    
        offsets_sel = offsets[mask]
        traces_sel = traces[mask]
    
        if positive_only:
            keep = offsets_sel >= 0
            offsets_sel = offsets_sel[keep]
            traces_sel = traces_sel[keep]
    
        order = np.argsort(offsets_sel)
        offsets_sel = offsets_sel[order]
        traces_sel = traces_sel[order]
    
        offsets_sel = offsets_sel[::step]
        traces_sel = traces_sel[::step]
    
        data_mat = traces_sel.T
        t = self.get_time_axis()
    
        return data_mat, offsets_sel, t

    def apply_linear_mute(self, data_mat, offsets, t, t_start=0.3, t_end=1.3):
        """
        Apply a linear top mute.
        
        Parameters
        ----------
        data_mat : ndarray
            Shape (nt, ntr)
        offsets : ndarray
            Shape (ntr,)
        t : ndarray
            Time axis in seconds
        t_start : float
            Mute time at smallest offset (seconds)
        t_end : float
            Mute time at largest offset (seconds)
        """
        out = data_mat.copy()
    
        x = offsets.astype(float)
        x0 = np.min(x)
        x1 = np.max(x)
    
        if x1 == x0:
            mute_t = np.full_like(x, t_start)
        else:
            mute_t = t_start + (t_end - t_start) * (x - x0) / (x1 - x0)
    
        for i in range(len(x)):
            out[t < mute_t[i], i] = 0.0
    
        return out, mute_t

    def read_segy_data(self, filename):
        with segyio.open(filename, "r", ignore_geometry=True) as f:
            data = f.trace.raw[:]
            dt = segyio.dt(f) / 1e6
    
            rec_list = []
    
            for i in range(f.tracecount):
                rec = geodata()
    
                rec.sx = f.header[i][segyio.TraceField.SourceX]
                rec.sy = f.header[i][segyio.TraceField.SourceY]
    
                rec.rx = f.header[i][segyio.TraceField.GroupX]
                rec.ry = f.header[i][segyio.TraceField.GroupY]
    
                rec.offset = np.sqrt((rec.rx - rec.sx)**2 + (rec.ry - rec.sy)**2)
    
                rec.delta_t = dt
                rec.data = np.asarray(data[i], dtype=float)
                rec.name = f"Trace {i}"
                rec.chan = i + 1
                rec_list.append(rec)   
    
        self.data = rec_list

    def apply_agc(self, method="rms"):
        """
        Apply AGC using:
        - 'rms'
        - 'max'
        - 'std'
        """
        gnew = SeismicGather()
        gnew.data = []
    
        for tr in self.data:
            trace = np.asarray(tr.data, dtype=float)
    
            if method == "rms":
                scale = np.sqrt(np.mean(trace**2))
            elif method == "max":
                scale = np.max(np.abs(trace))
            elif method == "std":
                scale = np.std(trace)
            else:
                raise ValueError("method must be 'rms', 'max', or 'std'")
    
            if scale == 0:
                scale = 1.0
    
            tr_new = tr.copy()
            tr_new.data = trace / scale
            gnew.data.append(tr_new)
    
        return gnew

    def copy(self):
       
        gnew = SeismicGather()
        gnew.data = [tr.copy() for tr in self.data]
    
        if hasattr(self, "dt"):
            gnew.dt = self.dt
        if hasattr(self, "sx"):
            gnew.sx = self.sx
        if hasattr(self, "sz"):
            gnew.sz = self.sz
    
        return gnew

    def surface_wave_dispersion(self, v_array=None):
    

        if v_array is None:
            v_array = np.linspace(100, 500, 50)
    
        freqs = np.fft.rfftfreq(len(self.data[0].data), self.data[0].delta_t)
    
        S_mat = []
    
        for vel in v_array:
            S = 0
    
            for tr in self.data:
                rfft = np.fft.rfft(tr.data)
                phase_shift = 2 * np.pi * tr.offset * freqs / vel
                S += rfft * np.exp(1j * phase_shift)
    
            S_mat.append(S)
    
        S_mat = np.abs(np.array(S_mat))
    
        # normalize each frequency band
        S_mat = S_mat / np.max(S_mat, axis=0)
    
        return freqs, v_array, S_mat
   
    def estimate_vs30_from_vr36(self, freqs, vels, S, wavelength=36.0, factor=1.08, fmin=1.0, fmax=20.0):
  
        freqs = np.asarray(freqs)
        vels = np.asarray(vels)
        S = np.asarray(S)
    
        # only use chosen frequency range
        fmask = (freqs >= fmin) & (freqs <= fmax)
    
        best_amp = -np.inf
        VR36 = None
        f_pick = None
    
        for f in freqs[fmask]:
    
            # for wavelength = 36 m: velocity = frequency * wavelength
            v_target = f * wavelength
    
            # skip if outside velocity range
            if v_target < np.min(vels) or v_target > np.max(vels):
                continue
    
            f_idx = np.argmin(np.abs(freqs - f))
            v_idx = np.argmin(np.abs(vels - v_target))
    
            amp = S[v_idx, f_idx]
    
            if amp > best_amp:
                best_amp = amp
                VR36 = vels[v_idx]
                f_pick = freqs[f_idx]
    
        Vs30 = factor * VR36
    
        print("VR36 =", VR36, "m/s")
        print("Pick frequency =", f_pick, "Hz")
        print("Vs30 =", Vs30, "m/s")
    
        return VR36, Vs30, f_pick

def get_receiver_gather(files, receiver_id=8):
   
    gather = SeismicGather()
    gather.data = []

    for filename in files:
        shot = SeismicGather()
        shot.read_segy_data(filename)

        if receiver_id < len(shot.data):
            tr = shot.data[receiver_id].copy()
            tr.name = filename
            gather.data.append(tr)

    return gather.sort_by_offset()