import numpy as np
import math
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# ============================================================
# --- CONSTANTS ---
# ============================================================
SAMPLE_RATE = 16000
CHANNELS = 4
FRAME_DURATION = 0.03   # 30ms
FRAMES_PER_ESTIMATE = 10
MIC_DISTANCE = 0.048
SPEED_OF_SOUND = 343.0
DEVICE = 'hw:3,0'

EMA_ALPHA = 0.3          # smoothing factor
MIN_CONF_VISUAL = 2.0    # minimum confidence to trust

# ============================================================
# --- BANDPASS FILTER ---
# ============================================================
def bandpass_filter(data, fs, lowcut=300, highcut=3000, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, data)

# ============================================================
# --- GCC-PHAT with confidence ---
# ============================================================
def gcc_phat(sig, refsig, fs=SAMPLE_RATE, interp=16):
    n = len(sig)
    sig /= np.linalg.norm(sig) + 1e-9
    refsig /= np.linalg.norm(refsig) + 1e-9

    window = np.hanning(n)
    SIG = np.fft.rfft(sig * window)
    REFSIG = np.fft.rfft(refsig * window)

    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-15

    cc = np.fft.irfft(R, n=interp * n)

    max_tau = MIC_DISTANCE / SPEED_OF_SOUND
    max_shift = int(interp * fs * max_tau)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    abs_cc = np.abs(cc)
    confidence = np.max(abs_cc) / (np.mean(abs_cc) + 1e-9)

    threshold = 0.6 * np.max(abs_cc)
    peaks = np.where(abs_cc > threshold)[0]

    if len(peaks):
        shift = peaks[np.argmin(np.abs(peaks - max_shift))] - max_shift
    else:
        shift = np.argmax(abs_cc) - max_shift

    tau = shift / (interp * fs)
    return np.clip(tau, -max_tau, max_tau), confidence

# ============================================================
# --- RECORD FRAME ---
# ============================================================
def record_frame():
    samples = int(SAMPLE_RATE * FRAME_DURATION)
    audio = sd.rec(samples, samplerate=SAMPLE_RATE,
                   channels=CHANNELS, device=DEVICE, dtype='float32')
    sd.wait()
    return audio

# ============================================================
# --- ESTIMATE RAW ANGLE ---
# ============================================================
def estimate_angle_multiframe():
    dxs, dys, confs = [], [], []

    for _ in range(FRAMES_PER_ESTIMATE):
        audio = record_frame()
        for ch in range(CHANNELS):
            audio[:, ch] = bandpass_filter(audio[:, ch], SAMPLE_RATE)

        m0, m1, m2, m3 = audio.T

        tau_01, c01 = gcc_phat(m0, m1)
        tau_12, c12 = gcc_phat(m1, m2)
        tau_23, c23 = gcc_phat(m2, m3)
        tau_30, c30 = gcc_phat(m3, m0)

        conf = np.mean([c01, c12, c23, c30])
        if conf < MIN_CONF_VISUAL:
            continue

        dx = (tau_01 - tau_23) * SPEED_OF_SOUND / 2
        dy = (tau_12 - tau_30) * SPEED_OF_SOUND / 2

        dxs.append(dx)
        dys.append(dy)
        confs.append(conf)

    if not dxs:
        return None, 0.0, None

    dx = np.average(dxs, weights=confs)
    dy = np.average(dys, weights=confs)
    confidence = np.mean(confs)

    raw_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    return raw_angle, confidence, (dx, dy)

# ============================================================
# --- POLAR PLOT SETUP ---
# ============================================================
plt.ion()
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='polar')

raw_point, = ax.plot([], [], 'bo', markersize=8, label='Raw')
smoothed_point, = ax.plot([], [], 'o', markersize=12, label='Smoothed')

ax.set_rmax(1)
ax.set_rticks([])
ax.set_title("Sound Direction (Raw & Smoothed)")
ax.legend(loc='upper right')

# ============================================================
# --- MAIN LOOP ---
# ============================================================
smoothed_x = None
smoothed_y = None

try:
    while True:
        raw_angle, confidence, vec = estimate_angle_multiframe()
        if raw_angle is None:
            print("No usable signal")
            continue

        # --- convert raw angle to radians ---
        angle_rad = np.deg2rad(raw_angle)

        # --- circular EMA smoothing ---
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)

        if smoothed_x is None:
            smoothed_x, smoothed_y = x, y
        else:
            smoothed_x = EMA_ALPHA * x + (1 - EMA_ALPHA) * smoothed_x
            smoothed_y = EMA_ALPHA * y + (1 - EMA_ALPHA) * smoothed_y

        smoothed_rad = np.arctan2(smoothed_y, smoothed_x)
        smoothed_angle = (np.degrees(smoothed_rad) + 360) % 360

        # --- print diagnostics ---
        print(f"Raw angle: {raw_angle:6.1f}°, Smoothed: {smoothed_angle:6.1f}°, Conf: {confidence:4.1f}")

        # --- update plot ---
        raw_point.set_data([angle_rad], [1])
        smoothed_point.set_data([smoothed_rad], [1])

        # Color-code smoothed point by confidence
        if confidence > 8:
            smoothed_point.set_color('red')
        elif confidence > 4:
            smoothed_point.set_color('orange')
        else:
            smoothed_point.set_color('gray')

        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stopped.")
