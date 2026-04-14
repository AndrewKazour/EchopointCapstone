# lidar_avoidance.py
# LiDAR-based obstacle avoidance for RPLidar + STM32 motor controller

from rplidar import RPLidar, RPLidarException
import serial
import time
import threading
import numpy as np

# ───────────────────────── SETTINGS ─────────────────────────

LIDAR_PORT      = '/dev/ttyUSB0'
MOTOR_PORT      = '/dev/ttyUSB0'   # Change to STM32 port when ready
BAUDRATE_LIDAR  = 115200
BAUDRATE_MOTOR  = 115200

ZONE_DANGER      = 1.00   # meters — triggers avoidance
MAX_RANGE_M      = 3.0
MIN_VALID_MM     = 150

FORWARD_CENTER   = 265   # degrees on sensor pointing forward on robot
FORWARD_HALF_ARC = 20    # ±20° cone = 40° total forward window

SIDE_ARC_START   = 40    # degrees offset from forward center
SIDE_ARC_END     = 120   # degrees offset from forward center

# Rear exclusion — ignore 120° behind the robot entirely
REAR_CENTER      = 85    # opposite of FORWARD_CENTER (265 - 180 = 85°)
REAR_HALF_ARC    = 60    # ±60° dead zone

SPEED_TURN       = 25    # turn speed percent sent to STM32

# Fixed turn duration — robot turns for this long then STOPS and re-checks.
# Tune this to roughly how long it takes your robot to turn ~45°.
TURN_DURATION    = 0.6   # seconds per turn burst

# After a turn burst, wait this long (stationary) before re-checking forward.
# Lets the LiDAR settle before deciding if path is clear.
SETTLE_TIME      = 0.4   # seconds

# Max number of turn bursts before giving up entirely.
MAX_TURN_BURSTS  = 6     # 6 x 0.6s = 3.6s max total turning

SCAN_HZ          = 10.0

# ─────────────────────────────────────────────────────────────


def send_motor(ser, forward_pct: int, turn_pct: int):
    if ser is None:
        print(f"  [MOTOR CMD] MOVE {forward_pct} {turn_pct}")
        return
    ser.write(f"MOVE {int(forward_pct)} {int(turn_pct)}\n".encode())


def send_stop(ser):
    if ser is None:
        print("  [MOTOR CMD] STOP")
        return
    ser.write(b"STOP\n")


def angle_in_arc(angle_deg, center, half_width):
    diff = (angle_deg - center + 180) % 360 - 180
    return abs(diff) <= half_width


def min_range_in_arc(scan, center_deg, half_width):
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
        and not angle_in_arc(angle, REAR_CENTER, REAR_HALF_ARC)
    ]
    return min(ranges) if ranges else float('inf')


def open_space_score(scan, center_deg, half_width):
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
        and not angle_in_arc(angle, REAR_CENTER, REAR_HALF_ARC)
        and dist_mm / 1000.0 <= MAX_RANGE_M
    ]
    return float(np.mean(ranges)) if ranges else MAX_RANGE_M


def choose_turn_direction(scan):
    """Returns +1 (right) or -1 (left) based on which side has more open space."""
    arc_hw       = (SIDE_ARC_END - SIDE_ARC_START) / 2
    arc_mid      = (SIDE_ARC_START + SIDE_ARC_END) / 2
    right_center = (FORWARD_CENTER + arc_mid) % 360
    left_center  = (FORWARD_CENTER - arc_mid) % 360

    right_score = open_space_score(scan, right_center, arc_hw)
    left_score  = open_space_score(scan, left_center,  arc_hw)

    print(f"  [TURN PICK] Left: {left_score:.2f} m  Right: {right_score:.2f} m")

    if right_score >= left_score:
        print("  → Choosing RIGHT (more open)")
        return 1
    else:
        print("  → Choosing LEFT (more open)")
        return -1


# ─────────────────────────────────────────────────────────────

class AvoidanceController:
    """
    Runs LiDAR scanning in a background thread.
    Sets self.in_control = True while actively avoiding.

    Mic code integration:
        if not avoidance.in_control:
            send_motor(motor_ser, forward_pct, turn_pct)
    """

    def __init__(self):
        self.in_control   = False
        self._stop_event  = threading.Event()
        self._thread      = None
        self._motor_ser   = None
        self._lock        = threading.Lock()
        self._latest_scan = None
        self._scan_lock   = threading.Lock()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("✅ Avoidance controller started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("🛑 Avoidance controller stopped")

    def _run(self):
        lidar = None
        try:
            # ── Motor serial ──────────────────────────────────
            # Testing without STM32: leave as None → commands just print
            # When STM32 is ready, comment out None and uncomment 3 lines below
            self._motor_ser = None
            # self._motor_ser = serial.Serial(MOTOR_PORT, BAUDRATE_MOTOR, timeout=0.1)
            # time.sleep(0.5)
            # print(f"✅ Motor UART open on {MOTOR_PORT}")

            lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE_LIDAR, timeout=2)
            time.sleep(1)
            print("✅ LiDAR connected:", lidar.get_info())
            print("   Health:", lidar.get_health())

            frame_interval = 1.0 / SCAN_HZ
            last_update    = time.time()

            while not self._stop_event.is_set():
                try:
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
                        print(f"[SCAN] Forward clear: {forward_dist:.3f} m")

                        if forward_dist <= ZONE_DANGER and not self.in_control:
                            threading.Thread(target=self._do_avoidance, daemon=True).start()

                except RPLidarException as e:
                    if "mismatch" in str(e).lower():
                        print("⚠️  Scan glitch, restarting scan loop...")
                        time.sleep(0.2)
                        continue
                    else:
                        raise

        except Exception as e:
            print("❌ Avoidance error:", e)
            raise
        finally:
            if self._motor_ser and self._motor_ser.is_open:
                send_stop(self._motor_ser)
                self._motor_ser.close()
            if lidar:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()

    def _do_avoidance(self):
        print("⚠️  OBSTACLE DETECTED — taking avoidance control")
        with self._lock:
            self.in_control = True

        # ── Step 1: Stop and assess ───────────────────────────
        send_stop(self._motor_ser)
        time.sleep(SETTLE_TIME)

        # ── Step 2: Pick turn direction once and commit ────────
        with self._scan_lock:
            scan_snapshot = self._latest_scan

        turn_dir = choose_turn_direction(scan_snapshot)
        dir_name = "RIGHT" if turn_dir > 0 else "LEFT"
        print(f"  Committed to turning {dir_name}")

        # ── Step 3: Turn in fixed bursts, check between bursts ─
        # Robot turns for TURN_DURATION, stops, waits SETTLE_TIME,
        # then checks if forward is clear before turning again.
        # This avoids spinning forever — each burst is deliberate.
        clear_threshold = ZONE_DANGER + 0.20

        for burst in range(MAX_TURN_BURSTS):
            if self._stop_event.is_set():
                break

            # Turn burst
            print(f"  [BURST {burst+1}/{MAX_TURN_BURSTS}] Turning {dir_name} for {TURN_DURATION}s")
            send_motor(self._motor_ser, 0, SPEED_TURN * turn_dir)
            time.sleep(TURN_DURATION)

            # Stop and settle
            send_stop(self._motor_ser)
            time.sleep(SETTLE_TIME)

            # Check forward
            with self._scan_lock:
                current_scan = self._latest_scan

            if current_scan is None:
                continue

            forward_dist = min_range_in_arc(current_scan, FORWARD_CENTER, FORWARD_HALF_ARC)
            print(f"  [CHECK] Forward: {forward_dist:.3f} m (need > {clear_threshold:.2f} m)")

            if forward_dist > clear_threshold:
                print(f"✅ Path clear ({forward_dist:.3f} m) after {burst+1} burst(s)")
                break
        else:
            print(f"⏱️  Max bursts reached — releasing control anyway")

        # ── Step 4: Release control ────────────────────────────
        send_stop(self._motor_ser)
        with self._lock:
            self.in_control = False
        print("🎤 Avoidance complete — control returned to microphone code")


# ─────────────────────────────────────────────────────────────

def main():
    controller = AvoidanceController()
    try:
        controller.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()