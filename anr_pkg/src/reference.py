import sys
import os
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped
import tf_transformations
import rclpy
import DR_init
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from scipy.spatial.transform import Rotation as R
import math
import time
from GripperControl import GripperControl
import open3d as o3d
import copy

rclpy.init()

ROBOT_ID   = "dsr01"
ROBOT_MODEL= ""

DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
node = rclpy.create_node('GlocalLabDemo', namespace=ROBOT_ID)
DR_init.__dsr__node = node

from DSR_ROBOT2 import print_ext_result, movej, movel, movec, move_periodic, move_spiral, set_velx, set_accx, DR_BASE, DR_TOOL, DR_AXIS_X, DR_MV_MOD_ABS
gripper = GripperControl()
Object_to_Camera = []

class RobotTransformer(Node):
    def __init__(self):
        super().__init__('grasp_transformer')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def get_robot_base_pose(self, camera_point):
        """
        YOLO로 얻은 물체 POSE를 Base 기준으로 바꾸는 함수

        return: [x,y,z] (mm)

        rtype: array
        """
        start_time = time.time()
        while time.time() - start_time < 0.5:
            rclpy.spin_once(self, timeout_sec=0.5)

        target_frame = 'base_link'
        source_frame = 'camera_link'

        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(seconds=0, nanoseconds=0),
                rclpy.duration.Duration(seconds=1.0)
            )

            # 카메라 좌표게 기준 POSE
            cx, cy, cz = camera_point
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

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
            # print("R", rotated_x, rotated_y, rotated_z)

            tx = t.transform.translation.x * 1000
            ty = t.transform.translation.y * 1000
            tz = t.transform.translation.z * 1000
            # print("tt", tx, ty, tz)

            final_x = tx + rotated_x
            final_y = ty + rotated_y
            final_z = tz + rotated_z

            return [final_x, final_y, final_z]

        except Exception as e:
            self.get_logger().error(f"수동 변환 실패: {e}")
            return None
        
transformer_node = RobotTransformer()
model = YOLO("YOLO_Model/workpiece1_OBB.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("Camera Detected")
profile = pipeline.start(config)
'''Depth 설정'''
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale() 
align_to = rs.stream.color
align = rs.align(align_to)

########################################  1. YOLO Detection  ########################################

def get_robust_depth(depth_frame, x, y, window_size = 5):
    """
    카메라를 통해 측정된 depth의 평균을 구하는 함수

    Returns:
        np.mean(valid_depths)
    """
    depth_data = np.asanyarray(depth_frame.get_data())
    half_w = window_size // 2
    
    y_start, y_end = max(0, int(y)-half_w), min(depth_data.shape[0], int(y)+half_w+1)
    x_start, x_end = max(0, int(x)-half_w), min(depth_data.shape[1], int(x)+half_w+1)
    
    roi = depth_data[y_start:y_end, x_start:x_end]

    valid_depths = roi[roi > 0]
    
    if len(valid_depths) > 0:
        return np.mean(valid_depths) * depth_scale
    else:
        return 0


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
                # print(f"현재 인식된 픽셀 좌표: px={px}, py={py}, dist={distance:.4f}m")
                
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
                
                best_target_this_frame = {
                    'pose': (-X_mm, -Y_mm, Z_mm, angle_deg)
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
    pipeline.stop()
    cv2.destroyAllWindows()

########################################  2. 로봇 좌표계로 변환  ########################################

if target_detected_pose:
    for _ in range(5):
        rclpy.spin_once(transformer_node, timeout_sec = 1.0)
    available_frames = transformer_node.tf_buffer._getFrameStrings()
    # print(f"Current Available TF List: {available_frames}")

    base_coords = transformer_node.get_robot_base_pose(target_detected_pose[:3])
    if base_coords:
                    target_detected_pose = {
                        'POS': base_coords,
                        'ANGLE': target_detected_pose[3] 
                    }
                    print(f"Robot Base Coordinate: {target_detected_pose['POS']}")
    else:
        print("No Transformation")
        sys.exit()

###############################  3. Multi-view Pose Estimation  ###############################

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

def Multiview_pcd_data(pipeline, align, T_base_camera, init_pose, save_dir, index, duration=1.0):
    """
    현재 카메라가 바라보고 있는 지점에서 정해진 시간(duration)동안 Point cloud를 수집하고 합친 이후 지정된 폴더에 저장하는 함수

    Returns:
        None
    """
    # -----------------------------
    # Parameters
    # -----------------------------
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
    while time.time() - start_time < duration:
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
        cv2.waitKey(1)

        # -----------------------------
        # Point cloud extraction
        # -----------------------------
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # m → mm
        verts *= 1000.0
        
        verts_h = np.c_[verts, np.ones(len(verts))]
        verts_base = (T_base_camera @ verts_h.T).T[:, :3]

        # object-centered distance
        init_pose = np.asarray(init_pose).reshape(3,)
        dist = np.linalg.norm(verts_base - init_pose[None, :], axis=1)

        valid = (dist < R_MAX)
        verts_valid = verts[valid]


        if verts_valid.shape[0] == 0:
            continue

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(verts_valid)
        pcd_tmp = pcd_tmp.voxel_down_sample(2.0)

        all_points.append(np.asarray(pcd_tmp.points))

    if len(all_points) == 0:
        print("No valid points collected.")
        return

    # -----------------------------
    # Merge & transform to base frame
    # -----------------------------
    merged_points = np.vstack(all_points)

    homog_points = np.hstack([merged_points,np.ones((merged_points.shape[0], 1))])

    base_points = (T_base_camera @ homog_points.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(base_points)
    pcd = pcd.voxel_down_sample(2.0)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0,origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord])

    # -----------------------------
    # Save
    # -----------------------------
    save_path = os.path.join(save_dir, f"view{index:02d}.pcd")
    o3d.io.write_point_cloud(save_path, pcd)

    tf_save_path = os.path.join(save_dir, f"view{index:02d}_tf.npy")
    np.save(tf_save_path, T_base_camera)

    print(f"Saved point cloud: {save_path}")
  
def get_current_camera_tf(transformer_node):
    """
    현재 로봇 Base 기준 카메라의 tf를 구하는 함수

    Returns:
        np.ndarray: Transform matrix (shape: (4,4))
    """
    try:
        t = transformer_node.tf_buffer.lookup_transform(
            'base_link',
            'camera_link_optical',
            rclpy.time.Time(), 
            rclpy.duration.Duration(seconds=2.0)
        )
        
        q = [t.transform.rotation.x, t.transform.rotation.y, 
             t.transform.rotation.z, t.transform.rotation.w]
        r = R.from_quat(q)
        rot_matrix = r.as_matrix()
        
        tx = t.transform.translation.x * 1000
        ty = t.transform.translation.y * 1000
        tz = t.transform.translation.z * 1000
        
        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = rot_matrix
        tf_matrix[:3, 3] = [tx, ty, tz]
        
        return tf_matrix
    
    except Exception as e:
        print(f"TF 조회 실패: {type(e).__name__} - {e}")
        return None

def get_rotation_matrix_z(deg):
    """
    입력된 각도(deg)만큼 z축으로 회전하는 행렬을 생성하는 함수

    Returns:
        np.ndarray: z축 회전 행렬 (shape: (3,3))
    """
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def merge_pcds_with_boundary(data_dir, initial_POS_YOLO, extent, initial_angle):
    """
    다각도에서 찍은 Point Cloud를 Bounding 박스로 자르고 하나로 합치는 함수

    Returns:
        Point cloud,  Bounding box
    """
    pcd_files = [f for f in os.listdir(data_dir) if f.endswith('.pcd') and "view0" in f]
    pcd_files.sort()
    merged_pcd = o3d.geometry.PointCloud()

    # OBB Box 회전 각도 확인 필요
    rotation_matrix = get_rotation_matrix_z(-initial_angle)
    # print("rotation matrix: ", rotation_matrix)
    center = np.array(initial_POS_YOLO)
    extent = np.array(extent)
    
    obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extent)
    obb.color = (1, 0, 0)

    for file_name in pcd_files:
        file_path = os.path.join(data_dir, file_name)
        pcd = o3d.io.read_point_cloud(file_path)
        #지면 제거
        plane_model, inliers = pcd.segment_plane(distance_threshold = 3.0, ransac_n = 3, num_iterations = 2000)
        [a, b, c, d] = plane_model
        pts = np.asarray(pcd.points)
        distances = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d
        removed_ground = np.where(abs(distances) > 1.5)[0]
        pcd = pcd.select_by_index(removed_ground)

        # BoundaryBox 제거
        pcd = pcd.crop(obb)
        
        merged_pcd += pcd
        print(f"Added {file_name}: {len(pcd.points)} Pointcloud Data being Merged")
        # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([merged_pcd, coord])

    # if not merged_pcd.is_empty():
    #     robot_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
    #     o3d.visualization.draw_geometries([merged_pcd, robot_base, obb], 
    #                                         window_name="Merged and Cropped PCD")  
        
    # BoundaryBox 제거
    # merged_pcd = merged_pcd.crop(obb)
    o3d.visualization.draw_geometries([merged_pcd, obb])
    return merged_pcd, obb

def preprocess_point_cloud_fps(pcd, num_points):
    """
    Point Cloud를 Downsampling, Normal estimation, Ransac을 위한 FPFH를 얻는 함수

    Returns:
        Point Cloud(Downsampled), FPFH
    """
    pcd_fps = pcd.farthest_point_down_sample(num_points)
    avg_dist = np.mean(pcd_fps.compute_nearest_neighbor_distance())
    
    radius_normal = avg_dist * 2
    # pcd_fps.orient_normals_towards_camera_location([ca])
    pcd_fps.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # normal vector를 z축으로 정렬
    normals = np.asarray(pcd_fps.normals)
    for i in range(len(normals)):
        if normals[i][2] < 0:   
            normals[i] *= -1

    radius_feature = avg_dist * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_fps,o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # o3d.visualization.draw_geometries([pcd_fps])

    return pcd_fps, pcd_fpfh

def RANSAC(source_down, target_down, source_fpfh, target_fpfh):
    max_attempts = 5
    attempt = 0
    results_history = []
    best_fitness = -0.1
    best_inlier_rmse = 100.0
    best_result = None
    best_threshold = None 

    while attempt < max_attempts:
        attempt += 1
        # thresholds = [10.0, 8.0, 6.0, 4.0, 2.0]
        # thresholds = [10.0, 9.0, 8.0, 7.0, 6.0]
        thresholds = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0]
        
        for thr in thresholds:            
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                thr,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(thr)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.99)
            )
            results_history.append({
                'threshold': thr,
                'fitness': result.fitness,
                'rmse': result.inlier_rmse
            })
            # temp_source = copy.deepcopy(source)
            # temp_source.transform(result.transformation)
            # o3d.visualization.draw_geometries([temp_source, target_down], window_name="RANSAC Result")

            if result.fitness > 0.85 and result.inlier_rmse < best_inlier_rmse:
                best_fitness = result.fitness
                best_inlier_rmse = result.inlier_rmse
                best_result = result
                best_threshold = thr
            if best_fitness > 0.95 and best_inlier_rmse < 2.35: 
                return best_result, best_threshold
            
        for res in results_history:
            marker = ">>" if res['fitness'] == best_fitness and res['rmse'] == best_inlier_rmse else "  "
            print(f"{marker} {res['threshold']:<9.2f} | {res['fitness']:<12.4f} | {res['rmse']:<12.4f}")
        print("="*55)
        print(f"Final Selection -> Fitness: {best_fitness:.4f}, RMSE: {best_inlier_rmse:.4f}")

    # if best_fitness > 0.90 and best_inlier_rmse < 2.4: 
    return best_result, best_threshold

def execute_icp_registration(source, target, transformation, best_threshold):
    # thresholds = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    multipliers = [1.0, 0.8, 0.5, 0.25]
    thresholds = [best_threshold * m for m in multipliers]
    best_fitness = -0.1
    best_inlier_rmse = 100.0
    results_history = []

    # voxel_size = 0.5
    # target.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    # source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    current_transformation = transformation

    for thr in thresholds:
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, thr, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), #PointToPlane
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1000) 
        )
        results_history.append({
            'threshold': thr,
            'fitness': reg_p2l.fitness,
            'rmse': reg_p2l.inlier_rmse
        })

        if reg_p2l.fitness > 0.85 and reg_p2l.inlier_rmse < best_inlier_rmse:
            best_fitness = reg_p2l.fitness
            best_inlier_rmse = reg_p2l.inlier_rmse
            best_result = reg_p2l

    for res in results_history:
        marker = ">>" if res['fitness'] == best_fitness and res['rmse'] == best_inlier_rmse else "  "
        print(f"{marker} {res['threshold']:<9.2f} | {res['fitness']:<12.4f} | {res['rmse']:<12.4f}")
    print("="*55)
    print(f"Final Selection -> Fitness: {best_fitness:.4f}, RMSE: {best_inlier_rmse:.4f}")

    return best_result

############### 과정3의 메인 루프 ############### 

try:
    pipeline.start(config)
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale() 
    align_to = rs.stream.color
    align = rs.align(align_to)

except RuntimeError:
    pass

if target_detected_pose:
    print("Multi-view Scanning Process")
    initial_POS = target_detected_pose['POS'] # mm
    # print("Initial Pos: {}".format(initial_POS)) 
    initial_angle = target_detected_pose['ANGLE'] # degree

    # Multi-view Scanning
    pcd_data_dir = "PCD_Data"
    start_time = time.time()
    while time.time() - start_time < 0.5:
        rclpy.spin_once(transformer_node, timeout_sec = 0.5)
    tf_base_to_cam = get_current_camera_tf(transformer_node)
    Multiview_pcd_data(pipeline, align, tf_base_to_cam, initial_POS, pcd_data_dir, 0, duration=1.0)  

    SCAN_HEIGHT = 400.0
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

    movel(circle_path[0], v=100, a=200)
    print("Scanning Start")

    for count, wp in enumerate(circle_path, 1):
        movel(wp, v=100, a=200)
        time.sleep(0.3)
        start_time = time.time()
        while time.time() - start_time < 0.5:
            rclpy.spin_once(transformer_node, timeout_sec=0.01)

        tf_base_to_cam = get_current_camera_tf(transformer_node)
        if tf_base_to_cam is not None:
            # print(tf_base_to_cam)

            Multiview_pcd_data(pipeline, align, tf_base_to_cam, initial_POS, pcd_data_dir, count, duration=1.0)
            obj_pos_base = np.array(initial_POS)
            T_base_object = np.eye(4)
            T_base_object[:3, 3] = obj_pos_base
            T_object_camera = np.linalg.inv(T_base_object) @ tf_base_to_cam
            camera_pos_object = T_object_camera[:3, 3]
            # print("camera pose object: {}".format(camera_pos_object))
            Object_to_Camera.append(camera_pos_object)        
        else:
            print(f"Warning: Failed to get TF for Viewpoint {count}")
        time.sleep(0.3)
        print(f"Scanned {count} Viewpoints")

    box_size = [65, 85, 60]
    # box_size = [1100, 1100, 1000]

    merged_pcd, boundary_box = merge_pcds_with_boundary(pcd_data_dir, initial_POS, box_size, initial_angle)
    if not merged_pcd.is_empty():
        o3d.io.write_point_cloud("PCD_Data/multiview_merged.pcd", merged_pcd)
    else:
        print("No Merged PCD")

    print("Downsampling in Process..")
    source = o3d.io.read_point_cloud("workpiece1/z_bracket_10000_rotated.pcd")
    target = o3d.io.read_point_cloud("PCD_Data/multiview_merged.pcd")
    source.paint_uniform_color([1.0, 0.0, 0.0])
    num_points = 10000
    num_points_target = 10000

    target_save_path = "PCD_Data/target_fps_downsampled.pcd"
    # if os.path.exists(target_save_path):
    if False:
        print(f"기존에 저장된 파일을 불러옵니다: {target_save_path}")
        target_down = o3d.io.read_point_cloud(target_save_path)
        avg_dist = np.mean(target_down.compute_nearest_neighbor_distance())
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 2, max_nn=30))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 5, max_nn=100))
    else:
        target_down, target_fpfh = preprocess_point_cloud_fps(target, num_points_target)
        o3d.io.write_point_cloud(target_save_path, target_down)
        print(f"연산 완료 및 저장됨: {target_save_path}")

    source_down, source_fpfh = preprocess_point_cloud_fps(source, num_points)

    print("RANSAC Process (Global Fitting)")
    target_down.paint_uniform_color([0, 0.651, 0.929])
    result_ransac, best_thr = RANSAC(source_down, target_down, source_fpfh, target_fpfh)

    source_temp = copy.deepcopy(source) 
    source_temp.transform(result_ransac.transformation)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_down,coord], window_name="RANSAC Result")

    print("ICP Pose Estimation")
    initial_trans = result_ransac.transformation
    result_icp = execute_icp_registration(source, target_down, initial_trans, best_thr)

    source_final = copy.deepcopy(source)
    source_final.transform(result_icp.transformation)

    source_final.paint_uniform_color([1, 0, 0])
    target_down.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([source_final, target_down, coord], window_name="Final ICP Result")

    initial_POS = result_icp.transformation[:3, 3]
    print("initial POS", initial_POS)
    initial_ORI = result_icp.transformation[:3, :3]
    print("initial Ori", initial_ORI)

####################################  4. Human Demonstration  ####################################

    Human_GraspingPose_Candidates = [
    [[-0.00294, -0.0285, 0.1741], [0.707107, -0.707107, 0, 0]], 
    [[-0.02584, -0.00947, 0.1513], [1, 0, 0, 0]], 
    [[0.02376, -0.00947, 0.151], [1, 0, 0, 0]], 
    [[0.00007, -0.00794, 0.1679], [0, -1, 0, 0]], 
    [[0.00007, 0.02866, 0.1295], [0, -1, 0, 0]], 
    ]

    Gripper_Gap_Size = [
        [57.8, 4.1],
        [57.8, 4.1],
        [57.8, 4.1],
        [85.0, 52.2],
        [85.0, 46.6],
    ]

    try:
        user_input = input("Press Appropriate Candidate: ")
        i = int(user_input)
        if 0 <= i < len(Human_GraspingPose_Candidates):
            offset_pos_local = np.array(Human_GraspingPose_Candidates[i][0])*1000
            offset_rot_quat = Human_GraspingPose_Candidates[i][1]
            approach_distance = 0.2
            approach_offset_local = np.array([0.0, 0.0, approach_distance * 1000])
            Gripper_Margin_local = np.array([0.0, 0.0, 0.065 * 1000])
            gripper_init, gripper_dis = Gripper_Gap_Size[i]

    except ValueError:
        print("Wrong Input")
        sys.exit()

    # Orinetation 반영
    # orientation = get_quaternion_from_euler(initial_angle)
    # rotation_ori_estimation = R.from_quat(orientation)
    rotation_ori_estimation = R.from_matrix(initial_ORI)
    r_offset = R.from_quat(offset_rot_quat)

    Robot_ROT = r_offset * rotation_ori_estimation

    # Position 반영 
    Robot_POS = initial_POS + rotation_ori_estimation.apply(offset_pos_local) + rotation_ori_estimation.apply(Gripper_Margin_local)
    Robot_POS_Approaching = Robot_POS + rotation_ori_estimation.apply(approach_offset_local)

    print(f"Calculation Finished")

########################################  5.Grasping  ########################################


if Robot_POS is not None and Robot_ROT is not None:
    print("Grasping Start")
    # Gripper와 Robot End-effector 거리 = 20cm + 1cm(margin)
    gripper.send_goal_by_gap_mm(gripper_init)

    Robot_ZYZ = Robot_ROT.as_euler('zyz', degrees=True) # Doosan M1509용 ZYZ

    robot_approaching = [
                Robot_POS_Approaching[0], Robot_POS_Approaching[1], Robot_POS_Approaching[2],
                Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]
    robot_grasping_pos = [
                Robot_POS[0], Robot_POS[1], Robot_POS[2],
                Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]
    robot_grasping_pos_go_up = [
            Robot_POS[0], Robot_POS[1], Robot_POS[2] + 100,
            Robot_ZYZ[0], Robot_ZYZ[1], Robot_ZYZ[2] ]

    movel(robot_approaching, v=30, a=60)
    time.sleep(1)
    print("Approaching")
    movel(robot_grasping_pos, v=20, a=40)
    gripper.send_goal_by_gap_mm(gripper_dis)
    print("Grasped")
    time.sleep(1)
    movel(robot_grasping_pos_go_up, v=20, a=40)
    time.sleep(3)
    gripper.send_goal(0.0)
