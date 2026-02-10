#!/usr/bin/env python3

import sys
import os
import math
import time
import copy

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO


import rospy
import tf2_ros
import tf_conversions  
from tf2_geometry_msgs import PointStamped


class RobotTransformer():
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)

    def get_robot_base_pose(self, camera_point):
        """
        A function that converts the object POSE obtained by YOLO to the base standard.

        return: [x,y,z] (mm)

        rtype: array
        """
        target_frame = 'base_0'
        source_frame = 'realsense_RGBframe'

        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )

            # Camera coordinates
            cx, cy, cz = camera_point
            
            # Quaternion from transform
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            # Manual Rotation Matrix calculation (as per your original code)
            r11 = 1 - 2*(qy**2 + qz**2)
            r12 = 2*(qx*qy - qz*qw)
            r13 = 2*(qx*qz + qy*qw)

            r21 = 2*(qx*qy + qz*qw)
            r22 = 1 - 2*(qx**2 + qz**2)
            r23 = 2*(qy*qz - qx*qw)

            r31 = 2*(qx*qz - qy*qw)
            r32 = 2*(qy*qz + qx*qw)
            r33 = 1 - 2*(qx**2 + qy**2)

            rotated_x = r11 * cx + r12 * cy + r13 * cz
            rotated_y = r21 * cx + r22 * cy + r23 * cz
            rotated_z = r31 * cx + r32 * cy + r33 * cz

            # Translation (converting meters to mm)
            tx = t.transform.translation.x * 1000
            ty = t.transform.translation.y * 1000
            tz = t.transform.translation.z * 1000

            final_x = tx + rotated_x
            final_y = ty + rotated_y
            final_z = tz + rotated_z

            return [final_x, final_y, final_z]

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF2 conversion failed: {e}")
            return None
        

def get_robust_depth(depth_frame, x, y, window_size = 5):
    """
    A function that calculates the average of the depth measured through the camera

    Returns:
        np.mean(valid_depths)
    """
    depth_data = np.asanyarray(depth_frame.get_data())
    half_w = window_size // 2
    
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]
    print(roi)

    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        return np.mean(valid_depths) * depth_scale
    else:
        return 0


def get_quaternion_from_euler(p_degree):
    """
    오일러각을 쿼터니안으로 바꾸는 함수

    Returns:
        [x, y, z, w]
    """
    theta = math.radians(p_degree)
    x = 0.0
    y = 0.0
    z = math.sin(theta / 2)
    w = math.cos(theta / 2)
    return [x, y, z, w]


def get_current_camera_tf(transformer_node):
    """
    A function that calculates the tf of the current robot base camera.

    Returns:
        np.ndarray: Transform matrix (shape: (4,4))
    """
    try:
        t = transformer_node.tf_buffer.lookup_transform(
            'base_0',
            'realsense_RGBframe',
            # 'camera_link_optical',
            rospy.Time(0), 
            rospy.Duration(2.0)
        )
        
        # Extract rotation using scipy (same as your ROS2 logic)
        q = [t.transform.rotation.x, t.transform.rotation.y, 
             t.transform.rotation.z, t.transform.rotation.w]
        r = R.from_quat(q)
        rot_matrix = r.as_matrix()
        
        # Translation: meters to millimeters
        tx = t.transform.translation.x * 1000
        ty = t.transform.translation.y * 1000
        tz = t.transform.translation.z * 1000
        
        # Build 4x4 Homogeneous Transformation Matrix
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        tf_matrix[:3, 3] = [tx, ty, tz]
        
        return tf_matrix
    
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF lookup failed: {e}")
        return None
    

def Multiview_pcd_data(pipeline, align, T_base_camera, init_pose, save_dir, index, duration=1.0):
    """
    A function that collects and merges point clouds from the point the camera is 
    currently looking at for a specified period of time and then saves them in a specified folder.

    Returns:
        None
    """
    # -----------------------------
    # Parameters
    # -----------------------------
    global points, verts, verts_h, verts_base, dist, valid, verts_valid, pcd_tmp, all_points
    R_MAX = 60.0                      # mm
    
    cam_pos = tf_base_to_cam[:3, 3]
    dist = np.linalg.norm(init_pose - cam_pos)
  
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)

    all_points = []
    start_time = time.time()
    count = 0

    # -----------------------------
    # Capture loop
    # -----------------------------
    # while time.time() - start_time < duration:
    while True:
        count += 1
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        if count % 3 != 0:
            continue

        depth_frame = decimation.process(depth_frame)

        # Visualization (optional)
        depth_color_frame = colorizer.colorize(depth_frame)
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        cv2.putText(depth_color_image,"Recording Case Test...",(30, 40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),2,)
        cv2.imshow("RealSense Depth - Scanning", depth_color_image)
        # cv2.waitKey(1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            print("Closing stream...")
            break

        # -----------------------------
        # Point cloud extraction
        # -----------------------------
        points = pc.calculate(depth_frame) # project depth frame to point cloud
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) #change to numpy array

        # m → mm
        verts *= 1000.0
        
        verts_h = np.c_[verts, np.ones(len(verts))] # change into something that can be multiplied
        # print(verts_h.shape)
        verts_base = (T_base_camera @ verts_h.T).T[:, :3]  # point cloudin base frame

        # # object-centered distance
        # init_pose = np.asarray(init_pose).reshape(3,)
        # dist = np.linalg.norm(verts_base - init_pose[None, :], axis=1)

        # valid = (dist < R_MAX)
        # verts_valid = verts[valid]

        # if verts_valid.shape[0] == 0:
        #     continue

        # pcd_tmp = o3d.geometry.PointCloud()
        # pcd_tmp.points = o3d.utility.Vector3dVector(verts_valid)
        # pcd_tmp = pcd_tmp.voxel_down_sample(2.0)

        # all_points.append(np.asarray(pcd_tmp.points))

        all_points.append(np.asarray(verts_base))


    cv2.waitKey(1)

    if len(all_points) == 0:
        print("No valid points collected.")
        return

    # -----------------------------
    # Merge & transform to base frame
    # -----------------------------
    merged_points = np.vstack(all_points)

    # homog_points = np.hstack([merged_points,np.ones((merged_points.shape[0], 1))])

    # base_points = (T_base_camera @ homog_points.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd = pcd.voxel_down_sample(2.0)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0,origin=[0, 0, 0])
    camframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0,origin=T_base_camera[:3, 3])
    initposeframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0,origin=init_pose)
    o3d.visualization.draw_geometries([pcd, coord, camframe, initposeframe])

    # -----------------------------
    # Save
    # -----------------------------
    save_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    o3d.io.write_point_cloud(save_path, pcd)

    tf_save_path = os.path.join(save_dir, f"view{index:02d}_tf.npy")
    np.save(tf_save_path, T_base_camera)

    print(f"Saved point cloud: {save_path}")


def get_look_at_zyz(camera_pos, target_pos, prev_zyz=None):
    """
    카메라 위치에서 목표점을 바라보도록 하는 orientation을 ZYZ Euler 각으로 계산하는 함수

    Returns:
        np.ndarray: ZYZ Euler angles in degrees
    """
    z_axis = np.array(target_pos) - np.array(camera_pos)
    z_axis = z_axis / (np.linalg.norm(z_axis)+ 1e-6)
    up = np.array([0, 0, 1])
    if abs(z_axis[2]) > 0.09:
        up = np.array([0, 1, 0])
        
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
    
    R_matrix = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(R_matrix)
    curr_zyz = r.as_euler('zyz', degrees=True)

    if prev_zyz is not None:
        new_zyz = np.copy(curr_zyz)
        for i in range(3):
            diff = new_zyz[i] - prev_zyz[i]
            if diff > 180: new_zyz[i] -= 360
            elif diff < -180: new_zyz[i] += 360
        return new_zyz
    return curr_zyz


################################### MAIN FUNCTION ###################################

if __name__ == "__main__":

    rospy.init_node('pose_estimation_node')
    transformer_node = RobotTransformer()
    Object_to_Camera = []


    ################################### Robot Initialization ###################################

    sys.dont_write_bytecode = True
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../doosan-robot/common/imp")))

    # for single robot (Use existing values ​​as is)
    ROBOT_ID    = "dsr01"
    ROBOT_MODEL = "m1013"

    # DSR ROBOT API
    import DR_init
    DR_init.__dsr__id    = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    from DSR_ROBOT import *


    ################################### RealSense Initialization ###################################

    model = YOLO("model/workpiece1_OBB.pt")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Camera Detected")
    profile = pipeline.start(config)

    # Depth setting
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    align_to = rs.stream.color
    align = rs.align(align_to)

    # print("start step 1")


    # ################################### YOLO OBB Detection ###################################

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            if not color_frame:
                continue
            best_target_this_frame = None

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, conf=0.92)
            annotated_frame = results[0].plot()
            if results[0].obb is not None:
                clss = results[0].obb.cls.cpu().numpy()
                confs = results[0].obb.conf.cpu().numpy()
                obb_boxes = results[0].obb.xywhr.cpu().numpy()
                up_indices = [i for i, c in enumerate(clss) if model.names[int(c)] == 'Up']            
                
                if len(up_indices) > 0:
                    up_confs = confs[up_indices]
                    max_sub_idx = np.argmax(up_confs)
                    max_idx = up_indices[max_sub_idx]
                    max_conf = confs[max_idx]
                    px, py, w, h, rotation = obb_boxes[max_idx]
                    
                    distance = get_robust_depth(depth_frame, px, py)
                    
                    camera_coords = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], distance)
                    # print(f"Intrinsics PPX: {intrinsics.ppx}, PPY: {intrinsics.ppy}")
                    X_mm, Y_mm, Z_mm = [c * 1000 for c in camera_coords]
                    angle_deg = np.degrees(rotation)

                    pose_text = f"X:{X_mm:.1f} Y:{Y_mm:.1f} Z:{Z_mm:.1f} Angle:{angle_deg:.1f}"
                    # print(pose_text)
                    cv2.circle(annotated_frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(annotated_frame, pose_text, (int(px) - 50, int(py) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    print(pose_text)
                    
                    best_target_this_frame = {
                        'pose': (X_mm, Y_mm, Z_mm, angle_deg)
                    }
                    cv2.circle(annotated_frame, (int(px), int(py)), 7, (0, 255, 255), -1) 
                    cv2.putText(annotated_frame, f"TOP SCORE: {max_conf:.2f}", (int(px)-50, int(py)-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow('Detection (Press K to Confirm)', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('k'):
                if best_target_this_frame is not None:
                    target_detected_pose = best_target_this_frame['pose']
                    # target_init_depth_yolo = target_detected_pose[2]
                    print(f"Pose extracted: X={target_detected_pose[0]:.1f}, Y={target_detected_pose[1]:.1f}, Z={target_detected_pose[2]:.1f}, Angle={target_detected_pose[3]:.1f}")
                    break 
                else:
                    print("No Detection")
                    sys.exit()
    finally:
        # pipeline.stop()
        cv2.destroyAllWindows()


    # ################################### Convert to robot coordinate system ###################################

    if target_detected_pose:
        available_frames = transformer_node.tf_buffer._getFrameStrings()
        print(f"Current Available TF List: {available_frames}")

        base_coords = transformer_node.get_robot_base_pose(target_detected_pose[:3])
        print(f"Base Coords: {base_coords}")
        if base_coords:
                        target_detected_pose = {
                            'POS': base_coords,
                            'ANGLE': target_detected_pose[3] 
                        }
                        print(f"Robot Base Coordinate: {target_detected_pose['POS']}")
        else:
            print("No Transformation")
            sys.exit()


    # ################################### Main loop of process 3 ################################### 

    if target_detected_pose:
        initial_POS = target_detected_pose['POS'] # mm
        initial_angle = target_detected_pose['ANGLE'] # degree
        pcd_data_dir = "PCD_Data"

        # start_time = time.time()
        # while time.time() - start_time < 0.5:
        #     rclpy.spin_once(transformer_node, timeout_sec = 0.5)
        tf_base_to_cam = get_current_camera_tf(transformer_node)


        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0,origin=[0, 0, 0])
        # camframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0,origin=tf_base_to_cam[:3, 3])
        # o3d.visualization.draw_geometries([coord, camframe])   


        Multiview_pcd_data(pipeline, align, tf_base_to_cam, initial_POS, pcd_data_dir, 0, duration=1.0)  
    
        SCAN_HEIGHT = 500.0
        DESIRED_ANGLE_DEG = 65.0
        rad_angle = math.radians(90.0 - DESIRED_ANGLE_DEG)
        SCAN_RADIUS = SCAN_HEIGHT * math.tan(rad_angle)
        NUM_POINTS = 4
        circle_path = []
        last_zyz = None

        for i in range(NUM_POINTS):
            angle = math.radians((360.0 / NUM_POINTS) * i)
            tx = initial_POS[0] + SCAN_RADIUS * math.cos(angle)
            ty = initial_POS[1] + SCAN_RADIUS * math.sin(angle)
            tz = initial_POS[2] + SCAN_HEIGHT
            curr_tcp_pos = [tx, ty, tz]

            zyz = get_look_at_zyz(curr_tcp_pos, initial_POS, last_zyz)
            circle_path.append([tx, ty, tz, zyz[0], zyz[1], zyz[2]])
            last_zyz = zyz

        # movel(circle_path[0], v=100, a=200)
        # print("Scanning Start")