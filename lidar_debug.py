from rplidar import RPLidar
import time

lidar = RPLidar('/dev/ttyUSB0', baudrate=115200, timeout=2)
time.sleep(1)
print("Place object in front, then move it to the left...")
for i, scan in enumerate(lidar.iter_scans()):
    if i < 2: continue
    if i > 15: break
    close = [(round(a), round(d/1000, 2)) for (_, a, d) in scan if 100 < d < 700]
    close.sort(key=lambda x: x[1])
    print(f"Scan {i} closest: {close[:5]}")
lidar.stop(); lidar.stop_motor(); lidar.disconnect()

