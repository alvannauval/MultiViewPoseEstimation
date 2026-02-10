#!/usr/bin/env python3

import rospy
import tf2_ros
import tf_conversions
import geometry_msgs.msg
import time
import os
import sys

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../doosan-robot/common/imp")))

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m1013"
import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
from DSR_ROBOT import *

def publish_tf():
    rospy.init_node('tf_robot_ft')

    broadcaster = tf2_ros.TransformBroadcaster()

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        ########## Publishing tf from End of robot to FT sensor
        robot_tf_stamped = geometry_msgs.msg.TransformStamped()
        robot_tf_stamped.header.stamp = rospy.Time.now()
        robot_tf_stamped.header.frame_id = "link6"
        robot_tf_stamped.child_frame_id = "ft_sensor_frame"
        robot_tf_stamped.transform.translation.x = 0.0
        robot_tf_stamped.transform.translation.y = 0.0
        robot_tf_stamped.transform.translation.z = 0.0375
        robot_tf_stamped.transform.rotation.x = 0.0
        robot_tf_stamped.transform.rotation.y = 0.0
        robot_tf_stamped.transform.rotation.z = -0.7071068
        robot_tf_stamped.transform.rotation.w =  0.7071068

        broadcaster.sendTransform(robot_tf_stamped)

        ########## Publishing tf from FT sensor to Gripper
        ft_gripper_stamped = geometry_msgs.msg.TransformStamped()
        ft_gripper_stamped.header.stamp = rospy.Time.now()
        ft_gripper_stamped.header.frame_id = "ft_sensor_frame"
        ft_gripper_stamped.child_frame_id = "gripper_frame"
        ft_gripper_stamped.transform.translation.x = 0.0
        ft_gripper_stamped.transform.translation.y = 0.0
        ft_gripper_stamped.transform.translation.z = 0.147 

        ft_gripper_stamped.transform.rotation.x = 0.0
        ft_gripper_stamped.transform.rotation.y = 0.0
        ft_gripper_stamped.transform.rotation.z = 0.0
        ft_gripper_stamped.transform.rotation.w = 1.0

        broadcaster.sendTransform(ft_gripper_stamped)

        # ########## Publishing tf from FT sensor to Realsense
        ft_realsense_stamped = geometry_msgs.msg.TransformStamped()
        ft_realsense_stamped.header.stamp = rospy.Time.now()
        ft_realsense_stamped.header.frame_id = "ft_sensor_frame"
        ft_realsense_stamped.child_frame_id = "camera_link"
        ft_realsense_stamped.transform.translation.x = -0.0175
        ft_realsense_stamped.transform.translation.y = -0.085150
        ft_realsense_stamped.transform.translation.z = 0.085350 

        ft_realsense_stamped.transform.rotation.x = 0.5
        ft_realsense_stamped.transform.rotation.y = -0.5
        ft_realsense_stamped.transform.rotation.z = 0.5
        ft_realsense_stamped.transform.rotation.w = 0.5

        broadcaster.sendTransform(ft_realsense_stamped)



        ########## Publishing tf from FT sensor to Intel Realsense
        ft_intel_stamped = geometry_msgs.msg.TransformStamped()
        ft_intel_stamped.header.stamp = rospy.Time.now()
        ft_intel_stamped.header.frame_id = "ft_sensor_frame"
        ft_intel_stamped.child_frame_id = "realsense_frame"
        ft_intel_stamped.transform.translation.x = 0.0115
        ft_intel_stamped.transform.translation.y = -0.085150
        ft_intel_stamped.transform.translation.z = 0.085350

        ft_intel_stamped.transform.rotation.x = 0.0
        ft_intel_stamped.transform.rotation.y = 0.0
        ft_intel_stamped.transform.rotation.z = 0.0
        ft_intel_stamped.transform.rotation.w = 1.0

        broadcaster.sendTransform(ft_intel_stamped)

        ########## Publishing Depth to RGB Camera tf
        intel_depth_rgb = geometry_msgs.msg.TransformStamped()
        intel_depth_rgb.header.stamp = rospy.Time.now()
        intel_depth_rgb.header.frame_id = "realsense_frame"
        intel_depth_rgb.child_frame_id = "realsense_RGBframe"
        intel_depth_rgb.transform.translation.x = -0.044
        intel_depth_rgb.transform.translation.y = 0.0
        intel_depth_rgb.transform.translation.z = 0.0
        intel_depth_rgb.transform.rotation.x = 0.0
        intel_depth_rgb.transform.rotation.y = 0.0
        intel_depth_rgb.transform.rotation.z = 0.0
        intel_depth_rgb.transform.rotation.w = 1.0

        broadcaster.sendTransform(intel_depth_rgb)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_tf()
    except rospy.ROSInterruptException:
        pass




# OLD
# import sys
# import rclpy
# from rclpy.node import Node
# import math
# from geometry_msgs.msg import TransformStamped
# from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
# import tf_transformations 

# class CameraStaticTfPublisher(Node):
#     def __init__(self):
#         super().__init__('camera_static_tf_publisher')
#         self.tf_static_broadcaster = StaticTransformBroadcaster(self)
#         self.make_static_transforms()

#     def make_static_transforms(self):
#         t = TransformStamped()

#         t.header.stamp = self.get_clock().now().to_msg()
#         t.header.frame_id = 'link_6'
#         t.child_frame_id = 'camera_link' #left ir camera -> robot frame    
#         t.transform.translation.x = -0.0325
#         t.transform.translation.y = -0.0595
#         t.transform.translation.z = 0.11525

#         quat = tf_transformations.quaternion_from_euler(0, 0, math.pi) # R, P, Y
#         t.transform.rotation.x = quat[0]
#         t.transform.rotation.y = quat[1]
#         t.transform.rotation.z = quat[2]
#         t.transform.rotation.w = quat[3]

#         opt = TransformStamped()
#         opt.header.frame_id = 'camera_link'
#         opt.child_frame_id = 'camera_link_optical' # -> left ir frame -> optical frame

#         opt.transform.translation.x = 0.0
#         opt.transform.translation.y = 0.0
#         opt.transform.translation.z = 0.0
#         q_opt = tf_transformations.quaternion_from_euler(0, 0, -math.pi) 
#         opt.transform.rotation.x = q_opt[0]
#         opt.transform.rotation.y = q_opt[1]
#         opt.transform.rotation.z = q_opt[2]
#         opt.transform.rotation.w = q_opt[3]
#         self.get_logger().info(f'Published static TF: {opt.header.frame_id} -> {opt.child_frame_id}')

#         self.tf_static_broadcaster.sendTransform([t,opt])

# def main():
#     rclpy.init()
#     node = CameraStaticTfPublisher()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()