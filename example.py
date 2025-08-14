import numpy as np
from pb_kinematics import PBKinematicsModel


#mocap_link_id = 20 # If using Fast-LIO 2
mocap_link_id = 17 # If using mocap or vive tracker

model = PBKinematicsModel("localhost", 6379, visualize=True, mocap_link_id=mocap_link_id)
while True:
    model.update_root_state(np.zeros(29))
    # When connected to real robot you can fuse IMU info to get smoother root orientation
    # model.update_root_state(np.zeros(29), imu_quat) # imu_quat: [x,y,z,w]