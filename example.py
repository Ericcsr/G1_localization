import numpy as np
from pb_kinematics import PBKinematicsModel



model = PBKinematicsModel("localhost", 6379, visualize=True)
while True:
    model.update_root_state(np.zeros(29))