import os
import pickle

import numpy as np
import pybullet as pb
import redis
from scipy.spatial.transform import Rotation

script_path = os.path.realpath(__file__)
current_file_directory = os.path.dirname(script_path)

REVOLUTE_JOINTS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
]

UPPER_BODY_JOINTS = REVOLUTE_JOINTS[12:]
WAIST_JOINTS = [13, 14, 15]

ZERO = [0.0] * 29
# @njit
# def set_joint_angles(robot_id: int, joint_indices: list[int], joint_angles: np.ndarray):
#     for i, joint_index in enumerate(joint_indices):
#         pb.resetJointState(robot_id, joint_index, joint_angles[i])


def get_root_pose_from_link(robot_id, link_id, joint_angles, target_link_pos, target_link_orn):
    """
    Given current joint angles and a target link's desired pose,
    compute the root frame pose.
    Parameters:
    robot_id (int): PyBullet ID of the robot.
    link_id (int): Index of the target link.
    joint_indices (list[int]): Joint indices to set.
    joint_angles (list[float]): Corresponding joint angles.
    target_link_pose (tuple): Desired pose of the link (position, quaternion).
    Returns:
    root_pose (tuple): Computed pose of the robot's root frame (position, quaternion).
    """
    # Reset joints to specified angles
    # pb.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
    current_root_pos, current_root_orn = pb.getBasePositionAndOrientation(robot_id)
    current_root_orn = Rotation.from_quat(current_root_orn)
    current_root_pos = np.array(current_root_pos)
    # set_joint_angles(robot_id, REVOLUTE_JOINTS, joint_angles)
    pb.resetJointStatesMultiDof(robot_id, REVOLUTE_JOINTS, targetValues=joint_angles.reshape(-1, 1))
    # Get the current pose of the link
    state = pb.getLinkState(robot_id, link_id, computeForwardKinematics=True)
    current_link_pos, current_link_orn = np.array(state[4]), np.array(state[5])
    current_link_pos = current_root_orn.inv().apply(current_link_pos - current_root_pos)
    current_link_orn = (current_root_orn.inv() * Rotation.from_quat(current_link_orn)).as_quat()
    # Convert poses to transformation matrices
    T_link_world = np.eye(4)
    T_link_world[:3, :3] = Rotation.from_quat(current_link_orn).as_matrix()
    T_link_world[:3, 3] = current_link_pos
    T_target_link = np.eye(4)
    T_target_link[:3, :3] = Rotation.from_quat(target_link_orn).as_matrix()
    T_target_link[:3, 3] = target_link_pos
    # Compute inverse transform to find root pose
    T_root_world = T_target_link @ np.linalg.inv(T_link_world)
    root_pos = T_root_world[:3, 3]
    root_orn = Rotation.from_matrix(T_root_world[:3, :3]).as_quat()
    # pb.resetBasePositionAndOrientation(robot_id, root_pos, root_orn)
    return root_pos, root_orn


def get_root_pose_from_link_simple(
    robot_id, link_id, joint_angles, target_link_pos, target_link_orn
):
    pb.resetJointStatesMultiDof(robot_id, [1, 2, 4], targetValues=joint_angles[::-1].reshape(-1, 1))
    state = pb.getLinkState(robot_id, link_id, computeForwardKinematics=True)
    root_pos_local, root_orn_local = np.array(state[0]), np.array(state[1])
    root_pos = target_link_pos + Rotation.from_quat(target_link_orn).apply(root_pos_local)
    root_orn = Rotation.from_quat(target_link_orn) * Rotation.from_quat(root_orn_local)
    return root_pos, root_orn.as_quat()


class QuaternionComplementaryFilter:
    """
    Complementary filter that fuses IMU into SLAM frame.
    Extrinsic q_ext = first_slam * inv(first_imu).
    """

    def __init__(self, tau, q_init=None):
        self.tau = float(tau)
        self.q = np.array([0.0, 0.0, 0.0, 1.0]) if q_init is None else self._normalize(q_init)
        self.q_ext = None  # IMU → SLAM

    def process_imu(self, q_imu_raw):
        q_imu = self._normalize(q_imu_raw)
        if self.q_ext is not None:
            # bring IMU into SLAM frame
            q_imu = self._quat_multiply(self.q_ext, q_imu)
        self.q = q_imu
        return self.q

    def process_slam(self, q_slam_raw, dt):
        q_slam = self._normalize(q_slam_raw)
        if self.q_ext is None:
            # first SLAM: compute IMU→SLAM extrinsic, snap to SLAM
            self.q_ext = self._quat_multiply(q_slam, self._quat_inverse(self.q))
            self.q = q_slam
            return self.q
        # thereafter just SLERP toward raw SLAM
        alpha = dt / (self.tau + dt)
        self.q = self._slerp(self.q, q_slam, alpha)
        return self.q

    def _normalize(self, q):
        q = np.asarray(q, float)
        return q / np.linalg.norm(q)

    def _quat_inverse(self, q):
        # unit quaternion inverse = conjugate
        return np.array([-q[0], -q[1], -q[2], q[3]])

    def _quat_multiply(self, a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    def _slerp(self, q0, q1, t):
        dot = np.dot(q0, q1)
        if dot < 0:
            q1, dot = -q1, -dot
        dot = np.clip(dot, -1, 1)
        theta = np.arccos(dot)
        if theta < 1e-6:
            return q0
        s = np.sin(theta)
        c0 = np.sin((1 - t) * theta) / s
        c1 = np.sin(t * theta) / s
        return (c0 * q0 + c1 * q1) / np.linalg.norm(c0 * q0 + c1 * q1)


class PBKinematicsModel:
    def __init__(self, redis_ip, redis_port, visualize=False, ref_robot=False, mocap_link_id=20):
        pb.connect(pb.GUI if visualize else pb.DIRECT)
        self.redis_client = redis.Redis(redis_ip, port=redis_port, db=0)
        self.robot = pb.loadURDF(f"{current_file_directory}/assets/g1/g1_29dof_kin.urdf")
        if ref_robot:
            self.ref_robot = pb.loadURDF(f"{current_file_directory}/assets/g1/g1_29dof_kin.urdf")
            for link_id in range(-1, pb.getNumJoints(self.ref_robot)):
                pb.changeVisualShape(self.ref_robot, link_id, rgbaColor=[0, 1, 0, 1])

        # self.simple_fk = pb.loadURDF(f"{current_file_directory}/assets/simple_fk_head.urdf")
        self.track_sites = [
            pb.loadURDF(f"{current_file_directory}/assets/frame.urdf") for _ in range(3)
        ]
        self.eef_id = [28, 36]
        self.head_id = 17
        self.waist_jid = [12, 13, 14]
        self.left_arm_jid = [15, 16, 17, 18, 19, 20, 21]
        self.right_arm_jid = [22, 23, 24, 25, 26, 27, 28]
        self.mocap_link_id = mocap_link_id
        self.root_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.quaternion_filter = QuaternionComplementaryFilter(tau=0.5)
        self.q = np.zeros(29)

    def set_robot_state(self, q, root_pose):
        pb.resetJointStatesMultiDof(self.robot, REVOLUTE_JOINTS, targetValues=q.reshape(-1, 1))
        pb.resetBasePositionAndOrientation(
            self.robot,
            root_pose[:3] + Rotation.from_quat(root_pose[3:]).apply(np.array([0.0, 0.0, -0.02])),
            root_pose[3:],
        )

    def set_ref_robot_state(self, q, root_pose):
        pb.resetJointStatesMultiDof(self.ref_robot, REVOLUTE_JOINTS, targetValues=q.reshape(-1, 1))
        pb.resetBasePositionAndOrientation(
            self.ref_robot,
            root_pose[:3] + Rotation.from_quat(root_pose[3:]).apply(np.array([0.0, 0.0, -0.02])),
            root_pose[3:],
        )

    def update_root_state(self, q, imu_quat=None):
        self.head_pos = pickle.loads(self.redis_client.get("head_pos"))  # type: ignore
        self.head_quat = pickle.loads(self.redis_client.get("head_quat"))  # type: ignore

        self.q = q
        root_pos, root_quat_slam = get_root_pose_from_link(
            self.robot, self.mocap_link_id, self.q, self.head_pos, self.head_quat
        )  # use full kinematic chain

        fused_quat = self.quaternion_filter.process_imu(imu_quat)
        fused_quat = self.quaternion_filter.process_slam(root_quat_slam, 0.02)
        pb.resetBasePositionAndOrientation(self.robot, root_pos, fused_quat)
        self.root_pose = np.concatenate((root_pos, fused_quat))
        return self.root_pose, root_quat_slam

    def get_opt_meta_pd(self, eef_stiffness, current_kp):
        """
        eef_stiffness: np.ndarray [3]
        """
        J_left = np.array(
            pb.calculateJacobian(
                self.robot,
                self.eef_id[0],
                localPosition=[0.0, 0.0, 0.0],
                objPositions=self.q.tolist(),
                objVelocities=ZERO,
                objAccelerations=ZERO,
            )[0]
        )[
            :, 6 + 12 :
        ]  # [3, 17]
        J_right = np.array(
            pb.calculateJacobian(
                self.robot,
                self.eef_id[1],
                localPosition=[0.0, 0.0, 0.0],
                objPositions=self.q.tolist(),
                objVelocities=ZERO,
                objAccelerations=ZERO,
            )[0]
        )[
            :, 6 + 12 :
        ]  # [3, 17]
        M = np.zeros((6, 17), dtype=np.float32)
        M[:3, :3] = J_left[:, :3]
        M[3:, :3] = J_right[:, :3]
        M[:3, 3:10] = J_left[:, 3:10]
        M[3:, 10:] = J_right[:, 10:]

        Kx = np.diag(np.repeat(1 / (eef_stiffness[:2] + 1e-8), 3))
        Kq = M.T @ Kx @ M
        # Implement PD regularization
        U, S, Vh = np.linalg.svd(M)
        V = Vh.T
        N = V[:, 6:]  # null space of M
        K_ref = np.diag(current_kp[12:])

        W0 = N.T @ (0.8*K_ref - Kq) @ N
        W = 0.5 * (W0 + W0.T)  # make sure W is symmetric
        Kq = Kq + N @ W @ N.T  # add null space regularization

        meta_pd_scale = Kq.diagonal() / current_kp[12:]

        print("unclipped meta pd:", meta_pd_scale)
        return meta_pd_scale.clip(min=0.5, max=1.5)

    def get_track_site(self):
        self.set_robot_state(self.q, self.root_pose)
        target = np.zeros((3, 7))
        left_hand = pb.getLinkState(self.robot, self.eef_id[0])
        right_hand = pb.getLinkState(self.robot, self.eef_id[1])
        head = pb.getLinkState(self.robot, 17)
        left_hand_pos, left_hand_quat = np.array(left_hand[0]), np.array(left_hand[1])
        right_hand_pos, right_hand_quat = np.array(right_hand[0]), np.array(right_hand[1])
        head_pos, head_quat = np.array(head[0]), np.array(head[1])
        target[0, :3] = left_hand_pos
        target[0, 3:] = left_hand_quat
        target[1, :3] = right_hand_pos
        target[1, 3:] = right_hand_quat
        target[2, :3] = head_pos
        target[2, 3:] = head_quat
        return target

    def visualize_track_sites(self, targets, only_pos=True):
        for i, frame in enumerate(self.track_sites):
            pb.resetBasePositionAndOrientation(
                frame, targets[i, :3], targets[i, 3:] if not only_pos else [0, 0, 0, 1]
            )

