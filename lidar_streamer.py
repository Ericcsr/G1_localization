#!/usr/bin/env python3
import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import redis
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

Z_OFFSET = 1.2 #  Roughly the height of the lidar above the ground in meters need finetuning


def gravity_aligned_transform(quat, pos, gravity_dir, target_down=np.array([0, 0, -1])):
    """
    Aligns a frame’s pose so that the supplied gravity direction
    maps to the `target_down` axis, and expresses the frame’s
    transform relative to that gravity-aligned origin.

    Parameters
    ----------
    quat : array‑like, shape (4,)
        The frame’s orientation as a quaternion [x, y, z, w]
        in the original world frame.
    pos : array‑like, shape (3,)
        The frame’s position in the original world frame.
    gravity_dir : array‑like, shape (3,)
        The gravity vector in the original world frame (e.g. [0, 0, -9.81]
        or any nonzero direction — only its orientation matters).
    target_down : array‑like, shape (3,), optional
        The desired “down” axis after alignment. Defaults to [0, 0, –1].

    Returns
    -------
    new_quat : ndarray, shape (4,)
        The frame’s orientation as a quaternion [x, y, z, w] after
        gravity alignment.
    new_pos : ndarray, shape (3,)
        The frame’s position after applying the gravity‑alignment
        rotation.

    Example
    -------
    >>> quat = [0, 0, 0, 1]      # identity
    >>> pos = [1, 2, 3]
    >>> g = [0.1, -0.2, -9.7]    # some noisy gravity reading
    >>> q2, p2 = gravity_aligned_transform(quat, pos, g)
    """
    # normalize inputs
    g = np.asarray(gravity_dir, float)
    g = g / np.linalg.norm(g)
    t = np.asarray(target_down, float)
    t = t / np.linalg.norm(t)

    # compute axis-angle to rotate g → t
    v = np.cross(g, t)
    s = np.linalg.norm(v)
    c = np.dot(g, t)

    if s < 1e-8:
        # gravity already aligned or opposite
        if c > 0:
            R_align = Rotation.identity()
        else:
            # 180° flip: pick any axis perpendicular to g
            # (here we choose an arbitrary one)
            ortho = np.array([1, 0, 0])
            if abs(g[0]) > 0.9:
                ortho = np.array([0, 1, 0])
            axis = np.cross(g, ortho)
            axis /= np.linalg.norm(axis)
            R_align = Rotation.from_rotvec(np.pi * axis)
    else:
        axis = v / s
        angle = np.arctan2(s, c)
        R_align = Rotation.from_rotvec(axis * angle)

    # apply alignment
    R_orig = Rotation.from_quat(quat)
    R_new = R_align * R_orig
    p_new = R_align.apply(pos)

    return R_new.as_quat(), p_new


class BodyTfListener(Node):
    def __init__(self):
        super().__init__("body_tf_listener")
        # Set up TF buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.imu_listener = self.create_subscription(Imu, "/livox/imu", self.imu_callback, 10)
        self.la_buffer = []
        self.g_rot = None
        # Create a 10 Hz timer; its callback will be queued and executed by spin_once()
        self.create_timer(0.01, self.on_timer)  # The while loop
        # pb.connect(pb.GUI)
        # self.frame = pb.loadURDF("../utils/assets/frame.urdf")
        self.rc = redis.Redis(host="localhost", port=6379, db=0)

    def update_slam_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "camera_init",  # target frame
                "body",  # source frame
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            t = trans.transform.translation
            q = trans.transform.rotation
            # self.get_logger().info(
            #     f"[BODY in CAMERA_INIT]  pos=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})  "
            #     f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
            # )

            lidar_pos = np.array([t.x, t.y, t.z])
            lidar_rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
            if self.g_rot is not None:
                self.lidar_rot, self.lidar_pos = gravity_aligned_transform(
                    lidar_rot.as_quat(), lidar_pos, self.g
                )
                self.lidar_pos[2] += Z_OFFSET
                self.rc.set("head_pos", pickle.dumps(self.lidar_pos))
                self.rc.set("head_quat", pickle.dumps(self.lidar_rot))
                # pb.resetBasePositionAndOrientation(self.frame, self.lidar_pos, self.lidar_rot)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def get_gravity_rotation(self, gravity_vector):
        # Compute the rotation that aligns the Z-axis with the gravity vector
        z_axis = gravity_vector / np.linalg.norm(gravity_vector)
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return Rotation.from_matrix(rotation_matrix)

    def on_timer(self):
        self.update_slam_pose()
        if len(self.la_buffer) == 10 and self.g_rot is None:
            self.g = -np.mean(self.la_buffer, axis=0)
            self.g_rot = self.get_gravity_rotation(self.g)

    def imu_callback(self, msg: Imu):
        self.la = msg.linear_acceleration
        if len(self.la_buffer) < 10:
            self.la_buffer.append(np.array([self.la.x, self.la.y, self.la.z]))


def main(args=None):
    rclpy.init(args=args)
    node = BodyTfListener()

    try:
        # Manually pump callbacks at ~10 Hz
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
