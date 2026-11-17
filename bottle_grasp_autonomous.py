import stretch_body.robot
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import random
import time
import sys
import os
import csv

# =======================================================
# --- 1. CONFIGURATION & TUNING ---
# =======================================================
CAM_WIDTH = 848      
CAM_HEIGHT = 480     
CAM_FPS = 30         

FRAME_CENTER_X = int(CAM_WIDTH / 2)
FRAME_CENTER_Y = int(CAM_HEIGHT / 2) # NEW: Vertical center target

RADIANS_PER_PIXEL = 0.0015   
LIFT_M_PER_PIXEL = 0.0005    # NEW: Safe proportional gain for lift height

# Grasping Distances
MACRO_STANDOFF_M = 0.45      
IDEAL_ARM_REACH_M = 0.35     
GRIPPER_OFFSET_M = 0.01      
TARGET_CLASS = 39            # YOLO class for 'bottle'

# =======================================================
# --- 2. DATA COLLECTION ENGINE (OPENVLA) ---
# =======================================================
def sleep_and_record(duration_sec, pipeline, align, robot, img_dir, csv_writer):
    """Pauses sequence logic while logging 10Hz data for OpenVLA."""
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        loop_start = time.time()
        
        # 1. Grab visual observation
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            continue
            
        img = np.asanyarray(color_frame.get_data())
        timestamp = time.time()
        img_filename = f"frame_{timestamp:.3f}.jpg"
        
        # Save image to disk
        cv2.imwrite(os.path.join(img_dir, img_filename), img)
        
        # 2. Grab physical robot state (Proprioception)
        status = robot.get_status()
        base_x = status['base']['x']
        base_theta = status['base']['theta']
        lift_pos = status['lift']['pos']
        arm_pos = status['arm']['pos']
        wrist_roll = status['end_of_arm']['wrist_roll']['pos']
        wrist_pitch = status['end_of_arm']['wrist_pitch']['pos']
        wrist_yaw = status['end_of_arm']['wrist_yaw']['pos']
        gripper_pos = status['end_of_arm']['stretch_gripper']['pos']
        
        # 3. Log the state-action pair to CSV
        instruction = "pick up the water bottle"
        csv_writer.writerow([
            timestamp, img_filename, instruction, 
            base_x, base_theta, lift_pos, arm_pos, 
            wrist_roll, wrist_pitch, wrist_yaw, gripper_pos
        ])
        
        # 4. Enforce ~10Hz logging rate (0.1 seconds per loop)
        elapsed = time.time() - loop_start
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)

# =======================================================
# --- 3. CAMERA & VISION UTILITIES ---
# =======================================================
def setup_wrist_camera():
    global FRAME_CENTER_X 
    global FRAME_CENTER_Y 
    ctx = rs.context()
    wrist_serial = None
    
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "D405" in name:
            wrist_serial = serial
            
    if not wrist_serial:
        raise RuntimeError("FATAL: Could not find D405 wrist camera.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(wrist_serial) 
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, )
    
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    
    # Update centers based on actual hardware resolution
    FRAME_CENTER_X = int(color_stream.width() / 2)
    FRAME_CENTER_Y = int(color_stream.height() / 2)
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    
    return pipeline, align, depth_scale

def flush_camera_buffer(pipeline, frames_to_drop=20):
    for _ in range(frames_to_drop):
        try:
            pipeline.wait_for_frames(timeout_ms=200)
        except RuntimeError:
            pass

def get_robust_depth(depth_image, x1, y1, x2, y2, depth_scale):
    width = x2 - x1
    height = y2 - y1
    min_x = int(x1 + (width * 0.25))
    max_x = int(x2 - (width * 0.25))
    min_y = int(y1 + (height * 0.25))
    max_y = int(y2 - (height * 0.25))
    
    depth_roi = depth_image[min_y:max_y, min_x:max_x]
    valid_depths = depth_roi[depth_roi > 0]
    
    if len(valid_depths) > 0:
        target_depth_raw = np.percentile(valid_depths, 15)
        return target_depth_raw * depth_scale 
    return None

def locate_target(pipeline, align, model, depth_scale, scan_time=5.0):
    timeout = time.time() + scan_time
    flush_camera_buffer(pipeline)
    
    while time.time() < timeout:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame, color_frame = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, conf=0.20, verbose=False)
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == TARGET_CLASS: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    target_dist_m = get_robust_depth(depth_image, x1, y1, x2, y2, depth_scale)
                    
                    target_cx = int((x1 + x2) / 2)
                    target_cy = int((y1 + y2) / 2) # NEW: Extract Y-center
                    
                    if target_dist_m:
                        return target_dist_m, target_cx, target_cy
    return None, None, None

# =======================================================
# --- 4. MAIN ROBOT SEQUENCE ---
# =======================================================
def execute_autonomous_grasp():
    # --- Dataset Folder Initialization ---
    episode_id = f"episode_{time.strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = os.path.join("openvla_dataset", episode_id)
    img_dir = os.path.join(dataset_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(dataset_dir, "trajectory.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'image_file', 'instruction', 'base_x', 'base_theta', 'lift_z', 'arm_extend', 'wrist_roll', 'wrist_pitch', 'wrist_yaw', 'gripper_pos'])

    # --- Hardware Initialization ---
    robot = stretch_body.robot.Robot()
    if not robot.startup():
        print("Failed to start robot.")
        sys.exit()
        
    model = YOLO("yolo11n.pt") 
    
    try:
        pipeline, align, depth_scale = setup_wrist_camera() 
    except Exception as e:
        print(e)
        sys.exit()

    try:
        # 1. Posture
        # Generate a random lift height between 0.65 and 0.85
        random_lift = random.uniform(0.65, 0.85)
        # Optional: Print the target height so you know what it chose
        print(f"Targeting lift height: {random_lift:.3f}") 
        print(f"\n--- Starting Data Collection: {episode_id} ---")
        print("[STEP 1] Moving to scan posture...")
        robot.lift.move_to(random_lift)
        robot.end_of_arm.move_to('stretch_gripper', 100) 
        robot.end_of_arm.move_to('wrist_pitch', 0)
        robot.end_of_arm.move_to('wrist_yaw', 0)
        robot.end_of_arm.move_to('wrist_roll', 0)
        robot.push_command()
        sleep_and_record(3.0, pipeline, align, robot, img_dir, csv_writer)

        # 2. Macro Scan
        print("\n[STEP 2] Macro Scan: Looking for bottle...")
        dist, cx, cy = locate_target(pipeline, align, model, depth_scale)

        if not dist:
            print("Aborting: Could not find bottle during Macro Scan.")
            return

        print(f" -> Target found at {dist:.2f}m. Centering base and lift...")
        pixel_error_x = cx - FRAME_CENTER_X 
        pixel_error_y = cy - FRAME_CENTER_Y 
        
        needs_centering = False
        
        if abs(pixel_error_x) > 40:
            robot.base.rotate_by(-(pixel_error_x * RADIANS_PER_PIXEL), v_r=0.1) 
            needs_centering = True
            
        if abs(pixel_error_y) > 30: # 30 pixel deadzone for height
            robot.lift.move_by(-(pixel_error_y * LIFT_M_PER_PIXEL), v_m=0.05)
            needs_centering = True
            
        if needs_centering:
            robot.push_command()
            sleep_and_record(3.0, pipeline, align, robot, img_dir, csv_writer)

        # 3. Macro Drive
        if dist > MACRO_STANDOFF_M:
            drive_step = dist - MACRO_STANDOFF_M
            print(f" -> Driving forward {drive_step:.2f}m to Standoff Position...")
            robot.base.translate_by(drive_step, v_m=0.05) 
            robot.push_command()
            sleep_and_record((drive_step / 0.05) + 1.5, pipeline, align, robot, img_dir, csv_writer)

        # 4. Micro Scan 
        print("\n[STEP 3] Micro Scan: Re-evaluating exact position...")
        dist, cx, cy = locate_target(pipeline, align, model, depth_scale, scan_time=3.0)

        if not dist:
            print("Aborting: Lost sight of the bottle during approach.")
            return

        print(f" -> Target locked perfectly at {dist:.2f}m. Final centering...")
        pixel_error_x = cx - FRAME_CENTER_X 
        pixel_error_y = cy - FRAME_CENTER_Y 
        
        needs_micro_centering = False
        
        if abs(pixel_error_x) > 20: 
            robot.base.rotate_by(-(pixel_error_x * RADIANS_PER_PIXEL), v_r=0.1)
            needs_micro_centering = True
            
        if abs(pixel_error_y) > 15: # Tighter tolerance on the micro scan
            robot.lift.move_by(-(pixel_error_y * LIFT_M_PER_PIXEL), v_m=0.05)
            needs_micro_centering = True
            
        if needs_micro_centering:
            robot.push_command()
            sleep_and_record(2.0, pipeline, align, robot, img_dir, csv_writer)

        # 5. Extend & Grasp
        target_extension = max(dist - GRIPPER_OFFSET_M, 0.05)
        print(f"\n[STEP 4] Executing Grasp! Extending arm {target_extension:.2f}m...")
        robot.arm.move_by(target_extension-0.13, v_m=0.04) 
        robot.push_command()
        sleep_and_record((target_extension / 0.04) + 1.0, pipeline, align, robot, img_dir, csv_writer)

        print(" -> Closing Gripper...")
        robot.end_of_arm.move_to('stretch_gripper', -50) 
        robot.push_command()
        sleep_and_record(2.0, pipeline, align, robot, img_dir, csv_writer)
        
        print(" -> Lifting...")
        robot.end_of_arm.move_to('wrist_roll', 1.57) 
        robot.lift.move_by(0.15)
        robot.push_command()
        sleep_and_record(2.0, pipeline, align, robot, img_dir, csv_writer)
            
        print("\nTask Complete. Data successfully saved for this episode.")
        
    finally:
        csv_file.close()
        pipeline.stop()
        robot.stop()

if __name__ == "__main__":
    try:
        execute_autonomous_grasp()
    except KeyboardInterrupt:
        print("\nAborting safely...")
        sys.exit(0)