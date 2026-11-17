# Autonomous Bottle Grasping on Hello Robot Stretch 3

Vision-guided, fully autonomous pick-up of a water bottle on a **Hello Robot Stretch 3** mobile manipulator, using **YOLO11** object detection and the **Intel RealSense D405** wrist camera's depth. Every run also records a synchronized image + robot-state trajectory at 10 Hz, structured for training vision-language-action models such as **OpenVLA**.

Built in the Neuro-symbolic Intelligent Cyber-physical Systems (NICE) Lab, Arizona State University (Prof. Zhe Xu).

| Wrist camera view | Depth + detection (macro scan) | Depth + detection (micro scan) |
|---|---|---|
| ![Wrist camera view of the bottle between the gripper fingers](docs/wrist_camera_view.jpg) | ![Depth map with YOLO bounding box during the macro scan](docs/depth_macro_scan.jpg) | ![Depth map with YOLO bounding box during the micro scan](docs/depth_micro_scan.jpg) |

---

## How it works

The script runs a coarse-to-fine, closed-loop grasp sequence:

1. **Scan posture.** The lift moves to a random height between 0.65 m and 0.85 m (so every recorded episode starts from a different viewpoint), the gripper opens and the wrist is straightened.
2. **Macro scan.** For up to 5 s, YOLO11 (`yolo11n.pt`) looks for a bottle (COCO class 39) in the D405 color stream. The distance to the bottle is the 15th percentile of valid depth values in the central 50% of its bounding box, which ignores background pixels and depth holes.
3. **Centering.** Horizontal pixel error rotates the base; vertical pixel error moves the lift. Each uses a proportional gain and a dead zone (40 px / 30 px).
4. **Approach.** If the bottle is farther than 0.45 m, the base translates by the difference.
5. **Micro scan.** A second, 3 s detection re-measures the bottle and re-centers with tighter dead zones (20 px / 15 px).
6. **Grasp.** The arm extends by the measured distance minus the camera-to-fingertip offset, the gripper closes, and the bottle is lifted 0.15 m.

```
scan posture ─► macro scan ─► center (base + lift) ─► approach ─► micro scan ─► re-center ─► extend ─► grasp ─► lift
      └──────────────────────── 10 Hz image + robot-state logging throughout ───────────────────────────┘
```

---

## Dataset recording (OpenVLA-style)

While the robot moves, frames and joint states are logged at about 10 Hz. Each run creates one episode folder:

```
openvla_dataset/
└── episode_YYYYMMDD_HHMMSS/
    ├── images/
    │   └── frame_<timestamp>.jpg
    └── trajectory.csv
```

`trajectory.csv` columns:

| Column | Meaning |
|---|---|
| `timestamp` | Unix time (s) |
| `image_file` | Matching image in `images/` |
| `instruction` | Language instruction: `pick up the water bottle` |
| `base_x`, `base_theta` | Base odometry (m, rad) |
| `lift_z` | Lift height (m) |
| `arm_extend` | Arm extension (m) |
| `wrist_roll`, `wrist_pitch`, `wrist_yaw` | Wrist joint angles (rad) |
| `gripper_pos` | Gripper position (stretch_body units) |

Actions for training can be derived as the difference between consecutive states.

---

## Requirements

**Hardware**
- Hello Robot Stretch 3 with the dexterous wrist and Stretch gripper
- Intel RealSense D405 wrist camera (detected automatically by name)

**Software**
- Python 3.8+
- [`stretch_body`](https://github.com/hello-robot/stretch_body) (preinstalled on Stretch)
- `pyrealsense2`, `ultralytics`, `opencv-python`, `numpy`

```bash
pip install pyrealsense2 ultralytics opencv-python numpy
```

`yolo11n.pt` downloads automatically on first run.

---

## Usage

1. Home the robot if it hasn't been homed since power-on:
   ```bash
   stretch_robot_home.py
   ```
2. Place a bottle within view of the wrist camera, clear the area around the robot, and keep the run-stop button within reach.
3. Run:
   ```bash
   python3 bottle_grasp_autonomous.py
   ```
4. Press `Ctrl+C` to abort. The camera, data file and robot are always shut down cleanly.

---

## Configuration

All tuning values are at the top of the script.

| Parameter | Default | Purpose |
|---|---|---|
| `CAM_WIDTH`, `CAM_HEIGHT` | 848 × 480 | D405 color resolution |
| `RADIANS_PER_PIXEL` | 0.0015 | Base-rotation gain for horizontal centering |
| `LIFT_M_PER_PIXEL` | 0.0005 | Lift gain for vertical centering |
| `MACRO_STANDOFF_M` | 0.45 | Distance to stop at before the micro scan |
| `GRIPPER_OFFSET_M` | 0.01 | Extra offset subtracted from the grasp reach |
| `TARGET_CLASS` | 39 | COCO class ID for "bottle" |
| YOLO confidence | 0.20 | Set in `locate_target()` |

---

## Limitations and next steps

- Detection, depth and gains were tuned for one lab setup; other lighting or bottle types may need retuning.
- Moves are timed with fixed waits rather than waiting for each joint to reach its target.
- There is no grasp-success check; a failed grasp still produces a complete-looking episode.
- Planned: label successful vs. failed episodes, log during scan phases, and fine-tune a policy on the collected episodes.

---

## Author

**Yashwanth Gowda**, MS Robotics and Autonomous Systems, Arizona State University
[LinkedIn](https://www.linkedin.com/in/nmyashwanthgowda) · [GitHub](https://github.com/yashwanthgowdanm)
