#!/usr/bin/env python3

import profile
import sys
import os
import math
import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO


def init_robot(robot_id="dsr01", model="m1013"):
    """Sets up the Doosan Robot API parameters."""
    DR_init.__dsr__id = robot_id
    DR_init.__dsr__model = model
    rospy.loginfo(f"Robot {robot_id} ({model}) initialized.")


def init_realsense():
    """Starts the RealSense pipeline and returns stream objects."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() # Usually 0.001 (1mm per unit)

    rospy.loginfo("RealSense Pipeline Started. Depth Scale is: {depth_scale}")
    return pipeline, align, intrinsics, depth_scale

def get_robust_depth(depth_frame, x, y, depth_scale, window_size=5):
    """
    Calculates the average depth in a window around (x, y) to avoid noise.
    """
    depth_data = np.asanyarray(depth_frame.get_data())
    half_w = window_size // 2
    
    # Define the bounding box for the window (ROI)
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]

    # Filter out zero values (invalid depth)
    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        # Return mean depth converted to meters
        return np.mean(valid_depths) * depth_scale
    else:
        return 0  # No valid depth found


def get_yolo_detection(pipeline, align, model, intrinsics, depth_scale):
    global results
    """Detects object via YOLO OBB and returns camera-space coordinates."""
    print("Waiting for YOLO detection... Press 'q' to confirm.")
    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_img = np.asanyarray(aligned.get_color_frame().get_data())
        
        results = model(color_img, conf=0.92)
        if results[0].obb is not None:
            # results[0].obb is sorted by confidence; index 0 is the best
            box = results[0].obb[0]
            px, py, _, _, rotation = box.xywhr.cpu().numpy()[0]
            
            # --- Robust Depth Calculation ---
            depth_frame = aligned.get_depth_frame()
            dist = get_robust_depth(depth_frame, px, py, depth_scale)
            
            if dist > 0:
                cam_pts = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], dist)
                cv_frame = results[0].plot()
                cv2.circle(cv_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.imshow("Detection (Press q)", cv_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return [c * 1000 for c in cam_pts], np.degrees(results[0].obb[0].xywhr.cpu().numpy()[0][4])
                

def calculate_look_at_zyz(camera_pos, target_pos):
    """Calculates ZYZ Euler angles to orient camera toward the target."""
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis /= (np.linalg.norm(z_axis) + 1e-6)
    
    up = np.array([0, 1, 0]) if abs(z_axis[2]) > 0.9 else np.array([0, 0, 1])
    x_axis = np.cross(up, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return R.from_matrix(rot_matrix).as_euler('zyz', degrees=True)


def get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe'):
    """Fetches the 4x4 Homogeneous Transformation matrix from TF2 (in mm)."""
    try:
        t = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(2.0))
        quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = R.from_quat(quat).as_matrix()
        tf_matrix[:3, 3] = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]) * 1000.0
        return tf_matrix
    except Exception as e:
        rospy.logerr(f"TF Lookup failed: {e}")
        return None
    

def capture_scan_view(pipeline, align, T_base_camera, index, save_dir="PCD_Data", duration=1.0):
    """Captures multiple frames over a duration and merges them into one clean PCD."""    
    all_points = [] # List to store points from every frame
    start_time = time.time()
    count = 0
    
    print(f"Scanning Viewpoint {index} for {duration}s...")

    while (time.time() - start_time) < duration:
        count += 1
        frame = pipeline.wait_for_frames()
        aligned_frames = align.process(frame)

        depth_frame = aligned_frames.get_depth_frame()
        last_depth_data = np.asanyarray(depth_frame.get_data())
        if not depth_frame:
            continue
            
        if count % 5 != 0:
            continue 

        # Calculate Points
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) * 1000.0
        
        verts_base = (T_base_camera @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        all_points.append(verts_base)

    if len(all_points) == 0:
        print("Error: No points captured!")
        return

    merged_verts = np.vstack(all_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_verts)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(last_depth_data, alpha=0.5), 
            cv2.COLORMAP_JET
        )
    img_file_path = os.path.join(save_dir, f"view{index:02d}_depth_color.png")
    cv2.imwrite(img_file_path, depth_colormap)

    pcd_file_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    tf_file_path = os.path.join(save_dir, f"view{index:02d}_tf.npy")
    np.save(tf_file_path, T_base_camera)
    o3d.io.write_point_cloud(pcd_file_path, pcd)
    print(f"Successfully saved merged {len(np.asarray(pcd.points))} points to {pcd_file_path}")


def home_robot():
    """Moves the robot to the predefined home joint position."""
    print("Moving to Home position...")
    movej([0, 0, 90, 0, 90, 0], v=15, a=30) 

if __name__ == "__main__":
    rospy.init_node('unified_grasp_scan')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    

    # Robot Initialization
    sys.dont_write_bytecode = True
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../doosan-robot/common/imp")))
    import DR_init
    init_robot()
    from DSR_ROBOT import *

    # RealSense Initialization
    pipeline, align, intrinsics, depth_scale = init_realsense()
    model = YOLO("model/workpiece1_OBB.pt")

    # Detection & Localization
    obj_cam_pos, cam_angle = get_yolo_detection(pipeline, align, model, intrinsics, depth_scale)
    T_init = get_tf_matrix(tf_buffer)
    obj_base_pos = (T_init @ np.append(obj_cam_pos, 1))[:3]
    obj_base_pose = [obj_base_pos[0], obj_base_pos[1], obj_base_pos[2], 0.0, cam_angle, 0.0]
    np.save("PCD_Data/initial_obj_pose.npy", obj_base_pose)


    # Scanning Parameters
    SCAN_HEIGHT = 500.0         # mm above the object
    VIEWPOINTS = 8              # Number of scans
    DESIRED_ANGLE_DEG = 75.0    # degrees
    pcd_save_dir = "PCD_Data"   
    
    SCAN_RADIUS = SCAN_HEIGHT * math.tan(math.radians(90.0 - DESIRED_ANGLE_DEG)) 

    # Multi View Scanning
    print(f"Starting Multi-view Scan...")
    print(f"Scanning from viewpoint no 1")
    capture_scan_view(pipeline, align, T_init, 0, save_dir=pcd_save_dir, duration=1.0)

    view_path = []

    for i in range(VIEWPOINTS):
        # Calculate circular position
        angle = math.radians((360.0 / VIEWPOINTS) * i)
        tx = obj_base_pos[0] + SCAN_RADIUS * math.cos(angle)
        ty = obj_base_pos[1] + SCAN_RADIUS * math.sin(angle)
        tz = obj_base_pos[2] + SCAN_HEIGHT
        
        # Calculate rotation so camera faces the object
        zyz = calculate_look_at_zyz([tx, ty, tz], obj_base_pos)
        
        # Combine into pose [x, y, z, a, b, c]
        target_pose = [tx, ty, tz, zyz[0], zyz[1], zyz[2]]

        view_path.append(target_pose)

    for i in range(VIEWPOINTS):
        print(f"Moving to Viewpoint {i}...")
        movel(view_path[i], v=100, a=200) # Doosan Move command
        time.sleep(1) 
        
        # Capture and merge from each viewpoint
        T_current = get_tf_matrix(tf_buffer)
        capture_scan_view(pipeline, align, T_current, i+1, save_dir=pcd_save_dir, duration=1.0)
        time.sleep(0.5)

    # Return Home
    home_robot()