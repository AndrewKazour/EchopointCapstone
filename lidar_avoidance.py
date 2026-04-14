# lidar_avoidance.py
# LiDAR-based obstacle avoidance for RPLidar
#
# Behavior:
#   1. Continuously scans for obstacles in the forward arc
#   2. On detection: STOPS, picks the clearer side, turns to clear, then resumes
#   3. Once the forward path is clear, sends STOP and releases motor control
#      so your microphone (voice command) code can take over
#
# Integration with your microphone code:
#   - This script exports `AvoidanceController` as a class
#   - Your mic code should check `controller.in_control` before sending MOVE commands
#   - When `in_control` is False, the avoidance system is idle and the path is clear
#   - Call `controller.start()` / `controller.stop()` to manage the thread

from rplidar import RPLidar
import serial
import math
import time
import threading
import numpy as np

# ───────────────────────── SETTINGS ─────────────────────────

LIDAR_PORT    = '/dev/ttyUSB0'
MOTOR_PORT    = '/dev/ttyUSB0'   # UART to STM32 — adjust to your wiring
BAUDRATE_LIDAR  = 115200
BAUDRATE_MOTOR  = 115200

# Danger zones (meters) — must match or be tighter than your STM32 expectations
ZONE_CRITICAL  = 0.50   # Stop immediately
ZONE_DANGER    = 1.00   # Begin avoidance
MAX_RANGE_M    = 3.0
MIN_VALID_MM   = 150

# Forward arc to watch for obstacles (degrees either side of 0°)
# 0° = directly ahead on your sensor mounting
FORWARD_HALF_ARC = 30   # watch ±30° → 60° cone in front

# Side arcs used to decide which way to turn
SIDE_ARC_START = 40     # degrees from front center
SIDE_ARC_END   = 120    # degrees from front center

# Motor command speeds (percent, matches STM32 MOVE protocol)
SPEED_FORWARD  = 35     # normal forward cruise
SPEED_TURN     = 25     # turning speed during avoidance
CLEAR_HYSTERESIS = 0.20 # extra margin added to ZONE_DANGER before declaring clear
SCAN_HZ        = 10.0   # target scan rate

# How long (seconds) to keep turning before re-checking the path
TURN_CHECK_INTERVAL = 0.4

# ─────────────────────────────────────────────────────────────


def send_motor(ser: serial.Serial, forward_pct: int, turn_pct: int):
    """Send a MOVE command to the STM32."""
    cmd = f"MOVE {int(forward_pct)} {int(turn_pct)}\n"
    ser.write(cmd.encode())


def send_stop(ser: serial.Serial):
    """Send a hard STOP to the STM32."""
    ser.write(b"STOP\n")


def angle_in_arc(angle_deg: float, center: float, half_width: float) -> bool:
    """True if angle_deg falls within [center - half_width, center + half_width] (mod 360)."""
    diff = (angle_deg - center + 180) % 360 - 180  # signed delta in (-180, 180]
    return abs(diff) <= half_width


def min_range_in_arc(scan, center_deg: float, half_width: float) -> float:
    """Return the closest valid reading (in meters) within the specified arc."""
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
    ]
    return min(ranges) if ranges else float('inf')


def open_space_score(scan, center_deg: float, half_width: float) -> float:
    """
    Higher score = more open space in that arc.
    Uses mean range — more robust than max alone.
    """
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
        and dist_mm / 1000.0 <= MAX_RANGE_M
    ]
    if not ranges:
        return MAX_RANGE_M  # unknown = assume open
    return float(np.median(ranges))


def choose_turn_direction(scan) -> int:
    """
    Returns +1 (turn right) or -1 (turn left) based on which side has more open space.
    Uses arcs at ±(SIDE_ARC_START..SIDE_ARC_END) degrees from forward (0°).
    """
    # Left side: 360 - SIDE_ARC_END … 360 - SIDE_ARC_START (since 0° is front)
    left_center  = 360 - ((SIDE_ARC_START + SIDE_ARC_END) / 2)
    right_center = (SIDE_ARC_START + SIDE_ARC_END) / 2
    arc_hw = (SIDE_ARC_END - SIDE_ARC_START) / 2

    left_score  = open_space_score(scan, left_center,  arc_hw)
    right_score = open_space_score(scan, right_center, arc_hw)

    print(f"  [TURN PICK] Left score: {left_score:.2f} m  Right score: {right_score:.2f} m")
    return -1 if right_score >= left_score else 1   # positive turn_pct = right on STM32


# ─────────────────────────────────────────────────────────────

class AvoidanceController:
    """
    Thread-safe obstacle avoidance controller.

    Your microphone / voice command code should:
        1. Check `controller.in_control` before sending its own MOVE commands.
        2. If True — avoidance owns the motors, mic code must wait.
        3. If False — avoidance is idle, mic code is free to drive.

    Example integration in your mic code:
        if not avoidance.in_control:
            send_motor(motor_ser, forward_pct, turn_pct)
    """

    def __init__(self):
        self.in_control = False       # True while avoidance is actively steering
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._motor_ser: serial.Serial | None = None
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────

    def start(self):
        """Start the avoidance loop in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("✅ Avoidance controller started")

    def stop(self):
        """Signal the avoidance loop to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("🛑 Avoidance controller stopped")

    # ── Internal loop ────────────────────────────────────────

    def _run(self):
        lidar = None
        try:
            self._motor_ser = serial.Serial(MOTOR_PORT, BAUDRATE_MOTOR, timeout=0.1)
            time.sleep(0.5)
            print(f"✅ Motor UART open on {MOTOR_PORT}")

            lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE_LIDAR, timeout=2)
            time.sleep(1)
            print("✅ LiDAR connected:", lidar.get_info())
            print("   Health:", lidar.get_health())

            frame_interval = 1.0 / SCAN_HZ
            last_update = time.time()
            last_scan = None

            for scan in lidar.iter_scans(max_buf_meas=500):
                if self._stop_event.is_set():
                    break

                now = time.time()
                if now - last_update < frame_interval:
                    continue
                last_update = now
                last_scan = scan

                forward_dist = min_range_in_arc(scan, 0.0, FORWARD_HALF_ARC)
                print(f"[SCAN] Forward clear: {forward_dist:.3f} m")

                if forward_dist <= ZONE_DANGER:
                    self._do_avoidance(lidar, scan)

                # After avoidance (or if no obstacle), make sure we're not holding motors
                if self.in_control:
                    self._release_motors()

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

    def _do_avoidance(self, lidar, first_scan):
        """
        Full avoidance sequence:
          1. Stop
          2. Pick turn direction from LiDAR data
          3. Turn until forward arc is clear
          4. Release motors → mic code can take over
        """
        print("⚠️  OBSTACLE DETECTED — taking avoidance control")
        with self._lock:
            self.in_control = True

        # ── Step 1: Stop ────────────────────────────────────
        send_stop(self._motor_ser)
        time.sleep(0.3)   # brief pause so robot fully stops before we assess

        # ── Step 2: Choose turn direction ───────────────────
        turn_dir = choose_turn_direction(first_scan)
        dir_name = "RIGHT" if turn_dir > 0 else "LEFT"
        print(f"  Turning {dir_name} to clear obstacle")

        # ── Step 3: Turn until clear ────────────────────────
        clear_threshold = ZONE_DANGER + CLEAR_HYSTERESIS

        for scan in lidar.iter_scans(max_buf_meas=500):
            if self._stop_event.is_set():
                send_stop(self._motor_ser)
                return

            forward_dist = min_range_in_arc(scan, 0.0, FORWARD_HALF_ARC)
            print(f"  [TURNING {dir_name}] Forward: {forward_dist:.3f} m  (need > {clear_threshold:.2f} m)")

            if forward_dist > clear_threshold:
                print(f"✅ Path clear ({forward_dist:.3f} m) — stopping turn")
                send_stop(self._motor_ser)
                break

            # Keep turning — no forward component while turning in place
            send_motor(self._motor_ser, 0, SPEED_TURN * turn_dir)
            time.sleep(TURN_CHECK_INTERVAL)

        # ── Step 4: Release control ──────────────────────────
        self._release_motors()

    def _release_motors(self):
        """Stop motors and hand control back to microphone code."""
        send_stop(self._motor_ser)
        with self._lock:
            self.in_control = False
        print("🎤 Avoidance complete — control returned to microphone code")


# Standalone run 

def main():
    controller = AvoidanceController()
    try:
        controller.start()
        # Block main thread — avoidance runs in background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
