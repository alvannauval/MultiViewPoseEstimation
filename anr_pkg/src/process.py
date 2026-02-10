import numpy as np
import open3d as o3d

frame = []

# Load the numpy file
tf_0 = np.load(r'TestingData\view00_tf.npy')
tf_1 = np.load(r'TestingData\view01_tf.npy')
tf_2 = np.load(r'TestingData\view02_tf.npy')
tf_3 = np.load(r'TestingData\view03_tf.npy')
tf_4 = np.load(r'TestingData\view04_tf.npy')
tf_obj = np.load(r'TestingData\initial_POS.npy')

# Load pcd files
pcd_0 = o3d.io.read_point_cloud(r"TestingData\view00.pcd")
pcd_1 = o3d.io.read_point_cloud(r"TestingData\view01.pcd")
pcd_2 = o3d.io.read_point_cloud(r"TestingData\view02.pcd")
pcd_3 = o3d.io.read_point_cloud(r"TestingData\view03.pcd")
pcd_4 = o3d.io.read_point_cloud(r"TestingData\view04.pcd")


world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])
world_frame_x = o3d.geometry.TriangleMesh.create_coordinate_frame(size=75.0, origin=[20, 0, 0])
world_frame_y = o3d.geometry.TriangleMesh.create_coordinate_frame(size=24.0, origin=[0, 20, 0])
frame.append(world_frame)
frame.append(world_frame_x)
frame.append(world_frame_y)

tf_0_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
tf_0_frame.transform(tf_0)
tf_1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
tf_1_frame.transform(tf_1)
tf_2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
tf_2_frame.transform(tf_2)
tf_3_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
tf_3_frame.transform(tf_3)
tf_4_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
tf_4_frame.transform(tf_4)
frame.append(tf_0_frame)
frame.append(tf_1_frame)
frame.append(tf_2_frame)
frame.append(tf_3_frame)
frame.append(tf_4_frame)

object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=tf_obj)
frame.append(object_frame)


o3d.visualization.draw_geometries(frame + [pcd_0])
