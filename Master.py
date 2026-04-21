import numpy as np
import math
import time
import sounddevice as sd
from scipy.signal import butter, sosfilt
import serial
import threading
from rplidar import RPLidar, RPLidarException

# ============================================================
# CONFIG - AUDIO
# ============================================================
SAMPLE_RATE         = 16000
CHANNELS            = 4
FRAME_DURATION      = 0.03
FRAMES_PER_ESTIMATE = 15

MIC_DISTANCE        = 0.048
SPEED_OF_SOUND      = 343.0
DEVICE              = 'hw:3,0'

EMA_ALPHA           = 0.3
ANGLE_OFFSET        = 0          # calibrate so 0 == robot forward
SOUND_THRESHOLD     = 0.004

# ============================================================
# CONFIG - SERIAL PORTS
# ============================================================
MOTOR_PORT    = '/dev/ttyS0'     # STM32 (GPIO UART or USB-serial)
LIDAR_PORT    = '/dev/ttyUSB0'   # RPLidar
BAUD_MOTOR    = 115200
BAUD_LIDAR    = 115200

# ============================================================
# CONFIG - SOUND TRACKING / MOTION
# ============================================================
FORWARD_ANGLE     = 0.0   # which GCC angle == straight ahead (calibrate this)
FORWARD_SPEED     = 100    # % forward speed when well aligned
TURN_GAIN         = 0.70  # turn_pct = angle_error * TURN_GAIN
MAX_TURN          = 85    # clamp turn to this %
ALIGN_THRESHOLD   = 35    # degrees -- within this, treat as aligned
FORWARD_DURATION  = 1.5   # seconds to drive forward per burst
TURN_BURST_DURATION = 0.6 # seconds to hold turn command before re-sampling

# FIX: Minimum PWM values that guarantee motors overcome static friction.
# If your robot still doesn't move, raise MIN_TURN_PCT in steps of 5
# until it just barely starts turning reliably.
MIN_TURN_PCT      = 50    # % — floor for turn commands (tune to your motors)
MIN_FWD_PCT       = 35    # % — floor for forward component while turning

# ============================================================
# CONFIG - LIDAR AVOIDANCE
# ============================================================
ZONE_CRITICAL     = 0.40  # metres -- stop immediately
ZONE_DANGER       = 0.50  # metres -- begin avoidance
MAX_RANGE_M       = 3.0
MIN_VALID_MM      = 150

FORWARD_CENTER    = 265   # sensor degrees that point forward on robot
FORWARD_HALF_ARC  = 35    # +-30 degree cone around forward (60 deg total)

SIDE_ARC_START    = 40    # degrees offset from forward for side scoring
SIDE_ARC_END      = 120

SPEED_TURN_AVOID  = 80    # % turn speed during avoidance
CLEAR_HYSTERESIS  = 0.20  # extra margin before declaring path clear
AVOID_FWD_TIME    = 1.2   # seconds to drive forward after clearing obstacle
SCAN_HZ           = 10.0

MAX_AVOID_TURN_TIME = 4.0  # seconds

# Peripheral blind-spot detection (wheel-clipping zones at ~135 and ~225 deg)
# These are sensor-frame angles, same reference as FORWARD_CENTER.
# The robot nudges away gently — no full stop, just a correction turn.
PERIPHERAL_RIGHT_CENTER  = 310  # sensor angle for right rear-side
PERIPHERAL_LEFT_CENTER   = 220  # sensor angle for left rear-side
PERIPHERAL_HALF_ARC      = 20    # +-20 deg cone per side (40 deg total each)
ZONE_PERIPHERAL          = 0.35  # metres -- trigger gentle nudge
SPEED_TURN_PERIPHERAL    = 25    # % turn speed for nudge (gentler than full avoid)
PERIPHERAL_NUDGE_TIME    = 0.4   # seconds to nudge before re-checking

# ============================================================
# BANDPASS FILTER  500-3000 Hz
# ============================================================
sos = butter(4, [500, 3000], btype='bandpass', fs=SAMPLE_RATE, output='sos')

# ============================================================
# MOTOR HELPERS
# ============================================================
_serial_lock = threading.Lock()

def send_move(ser, forward_pct, turn_pct):
    cmd = f"MOVE {int(forward_pct)} {int(turn_pct)}\n"
    with _serial_lock:
        ser.write(cmd.encode())
        ser.flush()

def send_stop(ser):
    with _serial_lock:
        ser.write(b"STOP\n")
        ser.flush()

# Shared audio energy updated each estimate_angle() call.
# The peripheral nudge reads this to decide which way to turn
# when both sides are equally clear.
_latest_audio_energy = [0.0, 0.0, 0.0, 0.0]  # one entry per channel
_audio_energy_lock   = threading.Lock()

# ============================================================
# GCC-PHAT
# ============================================================
def gcc(sig, ref, interp=8):
    n   = len(sig)
    sig = sig / (np.linalg.norm(sig) + 1e-9)
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    SIG = np.fft.rfft(sig)
    REF = np.fft.rfft(ref)
    R   = SIG * np.conj(REF)
    R  /= np.abs(R) + 1e-9
    cc  = np.fft.irfft(R, n * interp)
    max_tau   = MIC_DISTANCE / SPEED_OF_SOUND
    max_shift = int(interp * SAMPLE_RATE * max_tau)
    cc    = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    peak  = np.max(np.abs(cc))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau   = np.clip(shift / (interp * SAMPLE_RATE), -max_tau, max_tau)
    return tau, peak

def record():
    samples = int(SAMPLE_RATE * FRAME_DURATION)
    audio   = sd.rec(samples, samplerate=SAMPLE_RATE, channels=CHANNELS,
                     device=DEVICE, dtype='float32')
    sd.wait()
    return audio

def estimate_angle():
    global _latest_audio_energy
    vx_acc, vy_acc = 0.0, 0.0
    frames_used    = 0
    energy_acc     = [0.0, 0.0, 0.0, 0.0]
    pairs = [
        (0, 1,  1,  0),
        (1, 2,  0,  1),
        (2, 3, -1,  0),
        (3, 0,  0, -1),
    ]
    for _ in range(FRAMES_PER_ESTIMATE):
        audio = record()
        if np.mean(np.abs(audio)) < SOUND_THRESHOLD:
            continue
        for ch in range(CHANNELS):
            audio[:, ch] = sosfilt(sos, audio[:, ch])
            energy_acc[ch] += np.mean(np.abs(audio[:, ch]))
        m = audio.T
        vx = vy = 0.0
        for a, b, px, py in pairs:
            tau, q = gcc(m[a], m[b])
            d = tau * SPEED_OF_SOUND
            w = max(q, 0.01)
            vx += px * d * w
            vy += py * d * w
        vx_acc += vx
        vy_acc += vy
        frames_used += 1
    if frames_used == 0:
        return None
    # Update shared energy so peripheral nudge can read it
    with _audio_energy_lock:
        _latest_audio_energy = [e / frames_used for e in energy_acc]
    angle = math.degrees(math.atan2(vy_acc, vx_acc))
    return (angle + 360 + ANGLE_OFFSET) % 360

def signed_angle_error(detected, forward):
    return (detected - forward + 180) % 360 - 180

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

# ============================================================
# LIDAR HELPERS
# ============================================================
def angle_in_arc(angle_deg, center, half_width):
    diff = (angle_deg - center + 180) % 360 - 180
    return abs(diff) <= half_width

def min_range_in_arc(scan, center_deg, half_width):
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
    ]
    return min(ranges) if ranges else float('inf')

def open_space_score(scan, center_deg, half_width):
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
        and dist_mm / 1000.0 <= MAX_RANGE_M
    ]
    return float(np.mean(ranges)) if ranges else MAX_RANGE_M

def choose_turn_direction(scan):
    arc_hw       = (SIDE_ARC_END - SIDE_ARC_START) / 2
    arc_mid      = (SIDE_ARC_START + SIDE_ARC_END) / 2
    right_center = (FORWARD_CENTER + arc_mid) % 360
    left_center  = (FORWARD_CENTER - arc_mid) % 360
    right_score  = open_space_score(scan, right_center, arc_hw)
    left_score   = open_space_score(scan, left_center,  arc_hw)
    print(f"  [TURN PICK] Left={left_score:.2f}m  Right={right_score:.2f}m")
    if right_score >= left_score:
        print("  -> Choosing RIGHT")
        return 1
    print("  -> Choosing LEFT")
    return -1

# ============================================================
# LIDAR AVOIDANCE CONTROLLER
# ============================================================
class AvoidanceController:

    def __init__(self, motor_ser):
        self._ser         = motor_ser
        self.in_control   = False
        self._stop_event  = threading.Event()
        self._thread      = None
        self._lock        = threading.Lock()
        self._latest_scan = None
        self._scan_lock   = threading.Lock()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("LiDAR avoidance thread started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("LiDAR avoidance thread stopped")

    def _run(self):
        while not self._stop_event.is_set():
            lidar = None
            try:
                lidar = RPLidar(LIDAR_PORT, baudrate=BAUD_LIDAR, timeout=2)
                time.sleep(1)
                print("LiDAR connected:", lidar.get_info())
                print("  Health:", lidar.get_health())

                frame_interval = 1.0 / SCAN_HZ
                last_update    = time.time()

                for scan in lidar.iter_scans(max_buf_meas=500):
                    if self._stop_event.is_set():
                        break

                    with self._scan_lock:
                        self._latest_scan = scan

                    now = time.time()
                    if now - last_update < frame_interval:
                        continue
                    last_update = now

                    forward_dist = min_range_in_arc(scan, FORWARD_CENTER, FORWARD_HALF_ARC)
                    right_dist   = min_range_in_arc(scan, PERIPHERAL_RIGHT_CENTER, PERIPHERAL_HALF_ARC)
                    left_dist    = min_range_in_arc(scan, PERIPHERAL_LEFT_CENTER,  PERIPHERAL_HALF_ARC)
                    print(f"[LIDAR] fwd={forward_dist:.2f}m  "
                          f"side-R={right_dist:.2f}m  side-L={left_dist:.2f}m")

                    if forward_dist <= ZONE_DANGER:
                        with self._lock:
                            if self.in_control:
                                continue
                            self.in_control = True
                        threading.Thread(target=self._do_avoidance, daemon=True).start()

                    # Peripheral nudge: only fires if full avoidance is NOT active.
                    # Right-side obstacle → nudge left (turn_dir = -1).
                    # Left-side obstacle  → nudge right (turn_dir = +1).
                    elif right_dist <= ZONE_PERIPHERAL:
                        with self._lock:
                            if self.in_control:
                                continue
                            self.in_control = True
                        threading.Thread(
                            target=self._do_peripheral_nudge,
                            args=(-1, "RIGHT-SIDE"),
                            daemon=True
                        ).start()
                    elif left_dist <= ZONE_PERIPHERAL:
                        with self._lock:
                            if self.in_control:
                                continue
                            self.in_control = True
                        threading.Thread(
                            target=self._do_peripheral_nudge,
                            args=(1, "LEFT-SIDE"),
                            daemon=True
                        ).start()

            except RPLidarException as e:
                print(f"LiDAR exception (will retry): {e}")
            except Exception as e:
                print(f"LiDAR avoidance error (will retry): {e}")
            finally:
                if lidar:
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        lidar.disconnect()
                    except Exception:
                        pass
                if not self._stop_event.is_set():
                    print("  Reconnecting LiDAR in 0.5s ...")
                    time.sleep(0.5)

    def _do_peripheral_nudge(self, turn_dir, side_name):
        """Gentle correction when a wheel-clipping zone detects an object.
        No full stop — just a brief turn nudge to steer the body away,
        then immediately hand control back."""
        print(f"[PERIPHERAL] {side_name} obstacle -- nudging {'LEFT' if turn_dir > 0 else 'RIGHT'}")

        deadline = time.time() + PERIPHERAL_NUDGE_TIME
        while not self._stop_event.is_set() and time.time() < deadline:
            send_move(self._ser, 0, SPEED_TURN_PERIPHERAL * turn_dir)
            time.sleep(0.15)

        send_stop(self._ser)
        time.sleep(0.1)

        with self._lock:
            self.in_control = False
        print(f"[PERIPHERAL] nudge done -- resuming")

    def _do_avoidance(self):
        print("OBSTACLE DETECTED -- avoidance taking control")

        send_stop(self._ser)
        time.sleep(0.3)

        with self._scan_lock:
            snapshot = self._latest_scan
        turn_dir  = choose_turn_direction(snapshot)
        dir_name  = "RIGHT" if turn_dir < 0 else "LEFT"
        clear_thr = ZONE_DANGER + CLEAR_HYSTERESIS
        print(f"  Turning {dir_name} until forward > {clear_thr:.2f} m")

        deadline  = time.time() + MAX_AVOID_TURN_TIME
        timed_out = False

        while not self._stop_event.is_set():
            if time.time() >= deadline:
                print(f"  [WARN] Avoidance turn timed out after {MAX_AVOID_TURN_TIME}s -- proceeding anyway")
                timed_out = True
                send_stop(self._ser)
                break

            send_move(self._ser, 0, SPEED_TURN_AVOID * turn_dir)
            time.sleep(0.15)

            with self._scan_lock:
                current_scan = self._latest_scan
            if current_scan is None:
                continue

            forward_dist = min_range_in_arc(current_scan, FORWARD_CENTER, FORWARD_HALF_ARC)
            print(f"  [TURNING {dir_name}] forward={forward_dist:.2f}m  "
                  f"(need >{clear_thr:.2f}m)")

            if forward_dist > clear_thr:
                print(f"  Path clear ({forward_dist:.2f}m) -- stopping turn")
                send_stop(self._ser)
                break

        if not timed_out:
            print(f"  [3/3] Driving forward to clear obstacle ({AVOID_FWD_TIME}s)")
            deadline = time.time() + AVOID_FWD_TIME
            while time.time() < deadline:
                send_move(self._ser, 80, 0)
                time.sleep(0.15)

        send_stop(self._ser)
        time.sleep(0.5)   # let main loop settle before resuming

        with self._lock:
            self.in_control = False
        print("  Avoidance done -- mic code has control")


# ============================================================
# MAIN
# ============================================================
def main():
    print("Sound-tracking robot with LiDAR avoidance starting...")
    print(f"  Motor : {MOTOR_PORT} @ {BAUD_MOTOR}")
    print(f"  LiDAR : {LIDAR_PORT}")
    print(f"  Forward angle = {FORWARD_ANGLE} deg  (adjust ANGLE_OFFSET to calibrate)")
    print(f"  Motor deadband floors: turn>={MIN_TURN_PCT}%  fwd>={MIN_FWD_PCT}%")
    print()

    smoothed_x = smoothed_y = None

    motor_ser = serial.Serial(MOTOR_PORT, BAUD_MOTOR, timeout=1)
    time.sleep(2)
    ready_line = motor_ser.readline()
    if ready_line:
        print(f"STM32: {ready_line.decode(errors='replace').strip()}")

    avoidance = AvoidanceController(motor_ser)
    avoidance.start()

    try:
        while True:

            # ── 1. Estimate sound angle ──────────────────────────
            send_stop(motor_ser)
            time.sleep(0.12)
            raw_angle = estimate_angle()

            # ── 2. Wait out avoidance, then reset EMA ────────────
            if avoidance.in_control:
                while avoidance.in_control:
                    time.sleep(0.05)
                smoothed_x = smoothed_y = None
                continue

            if raw_angle is None:
                print("[no sound] -> STOP")
                send_stop(motor_ser)
                continue

            # ── 3. EMA smoothing on unit circle ─────────────────
            cx = math.cos(math.radians(raw_angle))
            cy = math.sin(math.radians(raw_angle))
            if smoothed_x is None:
                smoothed_x, smoothed_y = cx, cy
            else:
                smoothed_x = EMA_ALPHA * cx + (1 - EMA_ALPHA) * smoothed_x
                smoothed_y = EMA_ALPHA * cy + (1 - EMA_ALPHA) * smoothed_y

            angle = (math.degrees(math.atan2(smoothed_y, smoothed_x)) + 360) % 360

            # ── 4. P-controller ──────────────────────────────────
            error    = signed_angle_error(angle, FORWARD_ANGLE)

            # FIX: Apply minimum floors AFTER the P-controller so that
            # even a small computed turn_pct is boosted to at least
            # MIN_TURN_PCT — guaranteeing the motors overcome static
            # friction instead of humming in place.
            raw_turn = error * TURN_GAIN
            if raw_turn > 0:
                turn_pct = clamp(max(raw_turn, MIN_TURN_PCT), 0, MAX_TURN)
            elif raw_turn < 0:
                turn_pct = clamp(min(raw_turn, -MIN_TURN_PCT), -MAX_TURN, 0)
            else:
                turn_pct = 0.0

            if abs(error) <= ALIGN_THRESHOLD:
                fwd_pct = FORWARD_SPEED
            else:
                scale   = 1.0 - (abs(error) - ALIGN_THRESHOLD) / (180.0 - ALIGN_THRESHOLD)
                # FIX: Floor fwd_pct to MIN_FWD_PCT so the robot always
                # creeps forward while turning rather than pivoting on the
                # spot (which can stall motors on carpet/rough surfaces).
                fwd_pct = max(MIN_FWD_PCT, FORWARD_SPEED * scale)

            print(f"angle={angle:5.1f}  err={error:+6.1f}  "
                  f"fwd={fwd_pct:4.0f}%  turn={turn_pct:+5.0f}%")

            # ── 5. Hold command long enough for motors to act ────
            if abs(error) <= ALIGN_THRESHOLD:
                print(f"  [aligned] driving forward for {FORWARD_DURATION}s ...")
                deadline = time.time() + FORWARD_DURATION
                while time.time() < deadline:
                    if avoidance.in_control:
                        print("  [burst interrupted -- obstacle ahead]")
                        break
                    send_move(motor_ser, fwd_pct, turn_pct)
                    time.sleep(0.15)
            else:
                print(f"  [turning] holding command for {TURN_BURST_DURATION}s ...")
                deadline = time.time() + TURN_BURST_DURATION
                while time.time() < deadline:
                    if avoidance.in_control:
                        print("  [turn interrupted -- obstacle ahead]")
                        break
                    send_move(motor_ser, fwd_pct, turn_pct)
                    time.sleep(0.15)

            if not avoidance.in_control:
                send_stop(motor_ser)
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        avoidance.stop()
        send_stop(motor_ser)
        motor_ser.close()


if __name__ == "__main__":
    main()
