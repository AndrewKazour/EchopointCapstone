# lidar_avoidance.py
# LiDAR-based obstacle avoidance for RPLidar + STM32 motor controller

from rplidar import RPLidar
import serial
import time
import threading
import numpy as np

# ───────────────────────── SETTINGS ─────────────────────────

LIDAR_PORT      = '/dev/ttyUSB0'
MOTOR_PORT      = '/dev/ttyUSB0'   # Change when STM32 is connected
BAUDRATE_LIDAR  = 115200
BAUDRATE_MOTOR  = 115200

ZONE_CRITICAL    = 0.50   # Stop immediately
ZONE_DANGER      = 1.00   # Begin avoidance
MAX_RANGE_M      = 3.0
MIN_VALID_MM     = 150

FORWARD_HALF_ARC = 30     # ±30° cone in front
SIDE_ARC_START   = 40
SIDE_ARC_END     = 120

SPEED_TURN       = 25
CLEAR_HYSTERESIS = 0.20   # extra margin before declaring path clear
SCAN_HZ          = 10.0

# ─────────────────────────────────────────────────────────────


def send_motor(ser, forward_pct: int, turn_pct: int):
    if ser is None:
        print(f"  [MOTOR CMD] MOVE {forward_pct} {turn_pct}")
        return
    cmd = f"MOVE {int(forward_pct)} {int(turn_pct)}\n"
    ser.write(cmd.encode())


def send_stop(ser):
    if ser is None:
        print("  [MOTOR CMD] STOP")
        return
    ser.write(b"STOP\n")


def angle_in_arc(angle_deg: float, center: float, half_width: float) -> bool:
    diff = (angle_deg - center + 180) % 360 - 180
    return abs(diff) <= half_width


def min_range_in_arc(scan, center_deg: float, half_width: float) -> float:
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
    ]
    return min(ranges) if ranges else float('inf')


def open_space_score(scan, center_deg: float, half_width: float) -> float:
    ranges = [
        dist_mm / 1000.0
        for (_, angle, dist_mm) in scan
        if dist_mm >= MIN_VALID_MM
        and angle_in_arc(angle, center_deg, half_width)
        and dist_mm / 1000.0 <= MAX_RANGE_M
    ]
    if not ranges:
        return MAX_RANGE_M
    return float(np.mean(ranges))


def choose_turn_direction(scan) -> int:
    """
    Returns +1 (turn right) or -1 (turn left).
    Positive turn_pct = RIGHT on STM32.
    Picks the side with MORE open space.
    """
    arc_hw       = (SIDE_ARC_END - SIDE_ARC_START) / 2
    right_center = (SIDE_ARC_START + SIDE_ARC_END) / 2          # ~80°
    left_center  = 360 - ((SIDE_ARC_START + SIDE_ARC_END) / 2)  # ~280°

    right_score = open_space_score(scan, right_center, arc_hw)
    left_score  = open_space_score(scan, left_center,  arc_hw)

    print(f"  [TURN PICK] Left score: {left_score:.2f} m  Right score: {right_score:.2f} m")

    if right_score >= left_score:
        print("  → Choosing RIGHT (more open)")
        return 1
    else:
        print("  → Choosing LEFT (more open)")
        return -1


# ─────────────────────────────────────────────────────────────

class AvoidanceController:
    """
    Runs a continuous LiDAR scan in a background thread.
    Sets self.in_control = True while actively avoiding.
    Your mic code should check this flag before sending MOVE commands.
    """

    def __init__(self):
        self.in_control   = False
        self._stop_event  = threading.Event()
        self._thread      = None
        self._motor_ser   = None
        self._lock        = threading.Lock()

        # Shared scan buffer — main loop writes, avoidance turn loop reads
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
            # ── Motor serial ──────────────────────────────────────────────────
            # LiDAR-only test mode: motor_ser stays None, commands just print
            # When STM32 is ready, uncomment the three lines below and remove None
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

            for scan in lidar.iter_scans(max_buf_meas=500):
                if self._stop_event.is_set():
                    break

                # Always keep latest scan fresh — avoidance turn loop reads this
                with self._scan_lock:
                    self._latest_scan = scan

                now = time.time()
                if now - last_update < frame_interval:
                    continue
                last_update = now

                forward_dist = min_range_in_arc(scan, 0.0, FORWARD_HALF_ARC)
                print(f"[SCAN] Forward clear: {forward_dist:.3f} m")

                # Only trigger avoidance if not already avoiding
                if forward_dist <= ZONE_DANGER and not self.in_control:
                    self._do_avoidance()

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

        # ── Step 1: Stop ──────────────────────────────────────
        send_stop(self._motor_ser)
        time.sleep(0.3)

        # ── Step 2: Pick direction from latest scan ────────────
        with self._scan_lock:
            scan_snapshot = self._latest_scan

        turn_dir = choose_turn_direction(scan_snapshot)
        dir_name = "RIGHT" if turn_dir > 0 else "LEFT"
        print(f"  Turning {dir_name} to clear obstacle")

        # ── Step 3: Turn using rolling scan buffer ─────────────
        # _latest_scan is continuously updated by the main iter_scans loop
        # so we always check fresh data here without a second iterator
        clear_threshold = ZONE_DANGER + CLEAR_HYSTERESIS

        while not self._stop_event.is_set():
            time.sleep(0.15)

            with self._scan_lock:
                current_scan = self._latest_scan

            if current_scan is None:
                continue

            forward_dist = min_range_in_arc(current_scan, 0.0, FORWARD_HALF_ARC)
            print(f"  [TURNING {dir_name}] Forward: {forward_dist:.3f} m  (need > {clear_threshold:.2f} m)")

            if forward_dist > clear_threshold:
                print(f"✅ Path clear ({forward_dist:.3f} m) — stopping turn")
                send_stop(self._motor_ser)
                break

            send_motor(self._motor_ser, 0, SPEED_TURN * turn_dir)

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