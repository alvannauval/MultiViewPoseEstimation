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
import tf


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

        if aligned:
            print("Frames aligned successfully.")
        else:
            print("Failed to align frames.")
        
        results = model(color_img, conf=0.88)
        if results[0].obb is not None and len(results[0].obb) > 0:
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
                
        else:
            print("Searching for the object")
                

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

    
def capture_scan_view(pipeline, align, T_base_camera, index, save_dir="PCD_Data", duration=1.0):
    """
    Captures frames, merges PCD, and saves a side-by-side RGB+Depth visualization.
    Normalization ensures the depth image isn't just a solid blue block.
    """    
    all_points = []
    last_color_image = None
    last_depth_data = None
    start_time = time.time()
    count = 0
    
    print(f"Scanning Viewpoint {index} for {duration}s...")

    while (time.time() - start_time) < duration:
        count += 1
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        # Store for visualization (most recent frame)
        last_depth_data = np.asanyarray(depth_frame.get_data())
        last_color_image = np.asanyarray(color_frame.get_data())

        # Logic to reduce point density: process every 5th frame
        if count % 5 != 0:
            continue 

        # 1. Calculate Point Cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        # Convert to mm
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) * 1000.0
        
        # 2. Transform to Base Frame
        # T_base_camera must be the 4x4 matrix from important_2 logic
        verts_base = (T_base_camera @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
        all_points.append(verts_base)

    if len(all_points) == 0:
        print("Error: No data captured!")
        return

    # --- VISUALIZATION PROCESSING ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare RGB (BGR for OpenCV)
    # color_bgr = cv2.cvtColor(last_color_image, cv2.COLOR_RGB2BGR)

    # Prepare Normalized Depth (The "Blue Image" Fix)
    depth_mask = last_depth_data > 0
    if np.any(depth_mask):
        d_min = np.min(last_depth_data[depth_mask])
        d_max = np.max(last_depth_data[depth_mask])
        
        # Normalize to 0-255 range based on the distance of your object
        depth_norm = (last_depth_data - d_min) / (d_max - d_min + 1e-6)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        # Make invalid/reflective holes pure black
        depth_viz[~depth_mask] = [0, 0, 0]
    else:
        depth_viz = np.zeros_like(last_color_image)

    # Create Side-by-Side image
    side_by_side = np.hstack((last_color_image, depth_viz))
    
    # Save Image
    viz_path = os.path.join(save_dir, f"view{index:02d}_viz.png")
    cv2.imwrite(viz_path, side_by_side)

    # --- POINT CLOUD PROCESSING ---
    merged_verts = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_verts)
    
    # Voxel downsampling (2mm)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)

    # Save PCD and TF
    pcd_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    o3d.io.write_point_cloud(pcd_path, pcd)
    np.save(os.path.join(save_dir, f"view{index:02d}_tf.npy"), T_base_camera)

    print(f"Saved View {index}: {len(pcd.points)} points. Visualization: {viz_path}")


def pose_to_matrix(pose):
    """
    Converts [x, y, z, roll, pitch, yaw] in degrees to a 4x4 transformation matrix.
    Args:
    - pose (list or array): [x, y, z, roll, pitch, yaw] in degrees.
    
    Returns:
    - T (ndarray): The 4x4 homogeneous transformation matrix.
    """
    x, y, z = pose[:3]   # Translation vector
    roll, pitch, yaw = pose[3:]  # Rotation angles in degrees
    # 1. Create the 3x3 rotation matrix from roll, pitch, yaw (XYZ Euler Angles)
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    # 2. Build the 4x4 Homogeneous Transformation Matrix
    T = np.eye(4)  # Start with the identity matrix (4x4)
    T[:3, :3] = rot_matrix  # Set the upper-left 3x3 part to the rotation matrix
    T[:3, 3] = [x, y, z]    # Set the upper-right 3x1 part to the translation vector
    return T


def matrix_to_pose(matrix, zyz=False):
    """
    Converts a 4x4 homogeneous transform matrix into a 6D pose list.
    
    Args:
        matrix: 4x4 numpy array.
        zyz (bool): If True, returns [x, y, z, alpha, beta, gamma] using ZYZ order.
                   If False, returns [x, y, z, roll, pitch, yaw] using ZYX order.
    """
    # 1. Extract translation and rotation matrix
    x, y, z = matrix[:3, 3]
    rot_matrix = matrix[:3, :3]
    
    if isinstance(zyz, bool) and zyz:
        # Intrinsic ZYZ: R = Rz(alpha) * Ry(beta) * Rz(gamma)
        alpha, beta, gamma = R.from_matrix(rot_matrix).as_euler('ZYZ', degrees=True)
        return [x, y, z, alpha, beta, gamma]
    else:
        # Intrinsic ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        # SciPy returns [yaw, pitch, roll] for 'zyx'
        zyx = R.from_matrix(rot_matrix).as_euler('zyx', degrees=True)
        # Reorder to [x, y, z, roll, pitch, yaw]
        return [x, y, z, zyx[2], zyx[1], zyx[0]]


def get_tf_matrix(tf_buffer, target, source):
    """
    Gets the transformation matrix between two frames, with translation in millimeters.
    Args:
    - tf_buffer: tf2_ros.Buffer object
    - target (str): The name of the target frame
    - source (str): The name of the source frame
    Returns:
    - tf_matrix (numpy.ndarray): The 4x4 transformation matrix
      Translation is in millimeters, rotation as usual.
    """
    try:
        # 1. Try to get the transform at the EXACT current time
        # This waits up to 2.0s for the TF buffer to receive the robot's new position
        now = rospy.Time.now()
        tf_buffer.can_transform(target, source, now, rospy.Duration(2.0))
        transform = tf_buffer.lookup_transform(target, source, now)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        # 2. If the exact time fails, fallback to the latest available (Time 0)
        transform = tf_buffer.lookup_transform(target, source, rospy.Time(0), rospy.Duration(1.0))
        
    translation = transform.transform.translation
    rotation = transform.transform.rotation

    # Convert translation to millimeters
    translation_vector = np.array([translation.x, translation.y, translation.z]) * 1000.0
    
    # Convert quaternion to 4x4 rotation matrix
    # Using np.eye(4) ensures a clean matrix structure
    tf_matrix = tf.transformations.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
    
    # Insert translation into the 4x4 matrix
    tf_matrix[:3, 3] = translation_vector
    
    return tf_matrix


def transform_to_cam(t_base2goal, t_link2cam):
    """
    Correctly transforms the target from Camera Goal to Link6 Base.
    Input: 4x4 Homogeneous Matrix (T_base2goal)
    Output: [x, y, z, A, B, C] in Doosan ZYZ
    """
    # 1. Calculate the Link6 position in Base frame
    # T_base2link = T_base2goal @ T_cam2link
    # (Note: np.linalg.inv(T_link2cam) is T_cam2link)
    t_base2link = t_base2goal @ np.linalg.inv(t_link2cam)
    
    # 2. Convert directly to ZYZ for the Doosan Robot
    target_link6_zyz = matrix_to_pose(t_base2link, zyz==True)

    return target_link6_zyz


def zyz_to_rpy(zyz_angles, degrees=True):
    """
    Converts Euler ZYZ (Doosan style) to RPY (XYZ Euler).
    """
    # Create rotation object from ZYZ
    r = R.from_euler('zyz', zyz_angles, degrees=degrees)
    
    # Convert to RPY (extrinsic XYZ or intrinsic xyz depending on your transform_to_cam)
    # Most ROS-based 'RPY' uses 'xyz' (intrinsic) or 'XYZ' (extrinsic).
    rpy = r.as_euler('xyz', degrees=degrees)
    return rpy


def rpy_to_zyz(rpy_angles, degrees=True):
    """
    Converts RPY (XYZ Euler) to Euler ZYZ (Doosan style).
    RPY is usually interpreted as intrinsic xyz or fixed-axis XYZ.
    """
    # Create rotation object from RPY
    # Using 'xyz' (lowercase) denotes intrinsic rotations
    r = R.from_euler('xyz', rpy_angles, degrees=degrees)
    
    # Convert to ZYZ (intrinsic) for Doosan
    zyz = r.as_euler('zyz', degrees=degrees)
    return zyz


def home_robot():
    """Moves the robot to the predefined home joint position."""
    print("Moving to Home position...")
    movej([0, 0, 90, 0, 90, 0], v=15, a=30) 


def capture():
    T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
    time.sleep(1)
    capture_scan_view(pipeline, align, T_current, 0, save_dir=pcd_save_dir, duration=1.0)



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
    obj_cam_pos, obb_angle = get_yolo_detection(pipeline, align, model, intrinsics, depth_scale)
    T_init = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
    obj_base_pos = (T_init @ np.append(obj_cam_pos, 1))[:3]
    obj_base_pose = [obj_base_pos[0], obj_base_pos[1], obj_base_pos[2], 0.0, obb_angle, 0.0]
    np.save("PCD_Data/initial_obj_pose.npy", obj_base_pose)


    # Scanning Parameters
    SCAN_HEIGHT = 300           # m above the object
    VIEWPOINTS = 16             # Number of scans
    DESIRED_ANGLE_DEG = 65.0    # degrees
    pcd_save_dir = "PCD_Data"   
    
    SCAN_RADIUS = SCAN_HEIGHT * math.tan(math.radians(90.0 - DESIRED_ANGLE_DEG)) 

    # Multi View Scanning
    print(f"Starting Multi-view Scan...")
    print(f"Scanning from viewpoint no 1")
    capture_scan_view(pipeline, align, T_init, 0, save_dir=pcd_save_dir, duration=1.0)

    link6_path = []     # target based on link6
    camera_path = []    # target based on camera

    T_base2cam = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='base_0')
    T_link2cam = get_tf_matrix(tf_buffer, source='realsense_RGBframe', target='link6')    


    # Planning Phase
    for i in range(VIEWPOINTS):
        # Calculate circular position
        angle = math.radians((360.0 / VIEWPOINTS) * i)

        tx = obj_base_pose[0] + SCAN_RADIUS * math.cos(angle)
        ty = obj_base_pose[1] + SCAN_RADIUS * math.sin(angle)
        tz = obj_base_pose[2] + SCAN_HEIGHT

        zyz = calculate_look_at_zyz([tx, ty, tz], obj_base_pose[:3])
        rpy = zyz_to_rpy(zyz)

        link_pose_rpy = [tx, ty, tz, rpy[0], rpy[1], rpy[2]]
        link_pose_zyz = transform_to_cam(pose_to_matrix(link_pose_rpy), T_link2cam)
        link6_path.append(link_pose_zyz)

    # Moving Phase (PKL Thesis)
    if False:
        for i in range(VIEWPOINTS):
            print(f"Moving to Viewpoint {i+1}...")
            movel(link6_path[i], v=100, a=200) # Doosan Move command
            time.sleep(1) 

            # Capture and merge from each viewpoint
            T_current = get_tf_matrix(tf_buffer, target='base_0', source='realsense_RGBframe')
            capture_scan_view(pipeline, align, T_current, i+1, save_dir=pcd_save_dir, duration=1.0)
            time.sleep(0.5)

        # Return Home
        time.sleep(1)
        home_robot()


# # Test objet-camera-base transformation logic
# T_cam2ob = [obj_cam_pos[0], obj_cam_pos[1], obj_cam_pos[2], 0.0, 180.0, obb_angle] #T_cam2ob, xyzrpy def look outside
# T_base2ob = T_base2cam @ pose_to_matrix(T_cam2ob)
# T_ob2cam_goal = [[-1, 0, 0, 0],
#                  [0, 1, 0, 0],
#                  [0, 0, -1, SCAN_HEIGHT],
#                  [0, 0, 0, 1]]
# T_base2goal = T_base2ob @ T_ob2cam_goal
# matrix_to_pose(T_base2goal, zyz=True)
# # end of test