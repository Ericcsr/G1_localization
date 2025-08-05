# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import numpy as np
import sys
import time
import threading
from multiprocessing import Lock
from threading import Lock as TLock
from termcolor import cprint
from scipy.spatial.transform import Rotation as R

from mocap_utils.natnet_client import (
    NatNetClient,
    IN_FIELD_ROOM,
)

from multiprocessing.shared_memory import SharedMemory as SM
import pybullet as pb
import redis
from argparse import ArgumentParser
import pickle

RECENTER_OFFSET = np.array([0.23, -3.65, 0.0])


def y_up_to_z_up_quaternion(y_up_quat):
    """
    Converts a Y-up quaternion to a Z-up quaternion.

    Parameters:
        y_up_quat (array-like): Quaternion in Y-up frame [x, y, z, w].

    Returns:
        np.ndarray: Quaternion in Z-up frame [x, y, z, w].
    """
    # Convert the Y-up quaternion to a rotation object
    y_up_rotation = R.from_quat(y_up_quat)
    
    # Define a 90-degree rotation about the X-axis
    x_90_rotation = R.from_euler('x', 90, degrees=True)

    # z rotation

    z_rotation = R.from_euler('z', 90, degrees=True)
    
    # Apply the rotation to convert to Z-up
    z_up_rotation =  x_90_rotation * y_up_rotation * z_rotation
    
    # Return the Z-up quaternion
    return z_up_rotation.as_quat()

def weighted_average_quaternions(quaternions, weights):
    """
    Compute the weighted average of n quaternions using a fully vectorized NumPy approach.
    
    This function assumes the input quaternions are in [x, y, z, w] format.
    Internally, the algorithm converts them to [w, x, y, z] format, computes the weighted
    average via an eigen-decomposition, and then converts the result back to [x, y, z, w].
    
    Parameters
    ----------
    quaternions : array-like, shape (N, 4)
        An array of N quaternions in [x, y, z, w] format.
    weights : array-like, shape (N,)
        Weights corresponding to each quaternion (need not sum to 1).
    
    Returns
    -------
    avg_quat : ndarray, shape (4,)
        The weighted average quaternion in [x, y, z, w] format (unit norm).
    """
    # Convert inputs to numpy arrays
    quats = np.asarray(quaternions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).flatten()
    
    # Normalize weights to sum to 1
    weights /= np.sum(weights)
    
    # Convert quaternions from [x, y, z, w] to [w, x, y, z]
    # This is done by taking the last column as the scalar part.
    q_wxyz = np.concatenate((quats[:, 3:4], quats[:, :3]), axis=1)
    
    # Normalize each quaternion (each row)
    norms = np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    q_wxyz = q_wxyz / norms
    
    # Compute the weighted outer product sum:
    # A = sum_i weights[i] * (q_i * q_i^T)
    # Using einsum to avoid explicit Python loops.
    A = np.einsum('i,ij,ik->jk', weights, q_wxyz, q_wxyz)
    
    # Compute the eigen-decomposition of the symmetric matrix A.
    # The eigenvector corresponding to the largest eigenvalue is the average quaternion (in [w, x, y, z] format).
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    avg_q_wxyz = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Ensure the averaged quaternion is normalized.
    avg_q_wxyz /= np.linalg.norm(avg_q_wxyz)
    
    # Convert the averaged quaternion back from [w, x, y, z] to [x, y, z, w]
    avg_q_xyzw = np.array([avg_q_wxyz[1], avg_q_wxyz[2], avg_q_wxyz[3], avg_q_wxyz[0]])
    
    return avg_q_xyzw

def rotatepoint(q, v):
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[3:4]
    q_xyz = q[:3]
    t = 2*np.cross(q_xyz, v)
    return v + q_r * t + np.cross(q_xyz, t)

def get_heading(quat):
    ref_dir = np.zeros_like(quat[:3])
    ref_dir[0] = 1
    ref_dir = rotatepoint(quat, ref_dir)
    return np.arctan2(ref_dir[...,1], ref_dir[...,0])

class PBRobot:
    def __init__(self):
        self.r = pb.loadURDF("assets/g1/g1_29dof_rev_1_0.urdf", flags=pb.URDF_MERGE_FIXED_LINKS)

    def set_base_pose(self, pos, quat):
        pb.resetBasePositionAndOrientation(self.r, pos, quat)

class MocapQueue:
    def __init__(self, max_size, weight=[0.1,0.1,0.2,0.3,0.3]):
        assert len(weight) == max_size
        self.max_size = max_size
        self.weight = weight        
        self._pos_queue = []
        self._quat_queue = []

    def push(self, pos, quat):
        if len(self._pos_queue) == 0:
            self._pos_queue = self._pos_queue + [pos] * self.max_size
            self._quat_queue = self._quat_queue + [quat] * self.max_size
        elif len(self._pos_queue) == self.max_size:
            self._pos_queue = self._pos_queue[1:] + [pos]
            self._quat_queue = self._quat_queue[1:] + [quat]
            

    def filter(self, pos, quat):
        self.push(pos, quat)
        average_pos = np.average(np.array(self._pos_queue), axis=0, weights=self.weight)
        average_quat = weighted_average_quaternions(np.array(self._quat_queue), self.weight)
        return average_pos, average_quat

    def __len__(self):
        return len(self._pos_queue)

class MocapAgent(object):
    def __init__(
        self,
        ip=(
            "172.24.68.77" if IN_FIELD_ROOM else "172.24.68.19"
        ),  # NOTE: Domestic Suite or Dog Room
        use_multicast=True,
        focus_bot_name="G1_head",
        shared_memory=None,
        shared_memory_shm=None,
        ignore_hand_mocap_tracking_failure=True,
        optimize_for_single_robot=False,
        redis_client=None,
        use_relative = False
    ):
        self.ignore_hand_mocap_tracking_failure = ignore_hand_mocap_tracking_failure

        self._save_to_shm = False
        self.init = True
        self.use_relative = use_relative

        self.optimize_for_single_robot = optimize_for_single_robot
        self.redis_client = redis_client
        self.filter_queue = MocapQueue(5)
        self._focus_bot_name = focus_bot_name
        if shared_memory is not None:
            self._save_to_shm = True
            self._focus_bot_name = focus_bot_name
            self._shared_memory = shared_memory
            self._shared_memory_shm = shared_memory_shm

            self._shared_memory_lock = Lock()

        self._rigid_body_data = {}

        streaming_client = NatNetClient()
        streaming_client.set_client_address("127.0.0.1")
        streaming_client.set_server_address(ip)
        streaming_client.set_use_multicast(use_multicast)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.t_lock = TLock()
        self._last_time = time.time()
        streaming_client.new_frame_listener = self.receive_new_frame

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting 1")

        time.sleep(1)
        if streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting 2")

        self._last_time = time.time()
        self._thread = threading.Thread(target=self._check_time)
        self._thread.daemon = True
        self._thread.start()

        print("init done")

    def _check_time(self):
        while True:
            time_difference = time.time() - self._last_time
            #print(time_difference)
            if time_difference > 0.03:
                print("Time difference exceeded 0.03 seconds. Terminating program.")
                if self._save_to_shm:
                    topic_name = "base_mocap_pose"
                    array_shm = SM(name=topic_name)
                    array_shm.close()
                    cprint(f"closed existing shared memory for [{topic_name}].", "red", "on_yellow", attrs=["bold"])

                sys.exit(1)
            time.sleep(0.1)

    def receive_new_frame(self, data_dict):
        self._last_time = time.time()
        if IN_FIELD_ROOM:
            if self.optimize_for_single_robot:
                streaming_id_to_model_name = {
                    5: "G1_head",
                }
            else:
                assert False, "Not implemented for Field Room"
                streaming_id_to_model_name = {
                    1: "base",
                    2: "fingertips",
                    5: "G1_head",
                }
        else:
            raise NotImplementedError("Not implemented for Other Rooms")
        rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
        for i, rigid_body in enumerate(rigid_body_list):
            if self.optimize_for_single_robot:
                if rigid_body.id_num not in streaming_id_to_model_name.keys():
                    continue
                rigid_body_name = streaming_id_to_model_name[rigid_body.id_num]
                if rigid_body.tracking_valid:
                    if "G1_head" in rigid_body_name:
                        pos = np.array(rigid_body.pos)[[0,2,1]] * np.array([1, -1, 1]) + RECENTER_OFFSET
                        ori = y_up_to_z_up_quaternion(np.array(rigid_body.rot))
                        #pos, ori = self.filter_queue.filter(pose, ori)
                        pose = np.concatenate([pos, ori])

                    self.t_lock.acquire()
                    
                    if self.init:
                        self.init_pose = pose.copy()
                        self.init = False
                    if self.use_relative:
                        pose[:2] -= self.init_pose[:2]
                        heading = get_heading(self.init_pose[3:])
                        heading_inv = R.from_euler('z', -heading, degrees=False)
                        pose[:3] = heading_inv.apply(pose[:3])
                        pose[3:] = (heading_inv*R.from_quat(pose[3:])).as_quat()
                    self.redis_client.set("head_pos", pickle.dumps(pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(pose[3:]))
                    self._rigid_body_data[rigid_body_name] = pose
                    self.t_lock.release()
                else:
                    raise ValueError(f"Lost tracking for {rigid_body_name}")
            else:
                rigid_body_name = streaming_id_to_model_name[rigid_body.id_num]
                if rigid_body.tracking_valid:
                    if "G1_head" in rigid_body_name:
                        pos = np.array(rigid_body.pos)[[0,2,1]] * np.array([1, -1, 1]) + RECENTER_OFFSET
                        ori = y_up_to_z_up_quaternion(np.array(rigid_body.rot))
                        #pos, ori = self.filter_queue.filter(pose, ori)
                        pose = np.concatenate([pos, ori])
                    else:
                        # [x, y, z, thera_in_xy_plane]
                        robot_pos = rigid_body.pos
                        robot_ori = rigid_body.rot
                        pose = np.array(
                            [
                                robot_pos[2],
                                robot_pos[0],
                                robot_pos[1],
                                np.arctan2(robot_ori[1], robot_ori[3]) * 2,
                            ]
                        )
                        if not IN_FIELD_ROOM:
                            pose[-1] += np.pi   # NOTE: JUST FOR Dog Room
                        while pose[-1] < -np.pi:
                            pose[-1] += np.pi * 2
                        while pose[-1] > np.pi:
                            pose[-1] -= np.pi * 2

                    self.t_lock.acquire()
                    self._rigid_body_data[rigid_body_name] = pose
                    self.t_lock.release()
                else:
                    if not self.ignore_hand_mocap_tracking_failure:
                        if "hand" in rigid_body_name:
                            if rigid_body_name in self._rigid_body_data:
                                print(f"Lost tracking for {rigid_body_name}")
                                del self._rigid_body_data[rigid_body_name]

        if self._save_to_shm:
            self._shared_memory_lock.acquire()
            self._shared_memory[:] = self._rigid_body_data[self._focus_bot_name]
            self._shared_memory_lock.release()

    def rigid_body_data(self, key="G1_head"):
        self.t_lock.acquire()
        data = self._rigid_body_data[key].copy()
        self.t_lock.release()
        return data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--use_relative", action="store_true", default=False)
    args = parser.parse_args()
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    agent = MocapAgent(optimize_for_single_robot=True, redis_client=redis_client, use_relative = args.use_relative)
    if args.visualize:
        pb.connect(pb.GUI)
        robot = PBRobot()
    
    ts = time.time()
    init = True
    while True:
        rigid_body_data = agent.rigid_body_data()
        pos, quat = rigid_body_data[:3], rigid_body_data[3:]
        #pos, quat = agent.filter_queue.filter(pos, quat)
        if args.visualize:
            robot.set_base_pose(pos, quat)
        time.sleep(1.0 / 120)