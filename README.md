# G1 Localization 3 in 1 solution
## FAST-LIO2
### Install ROS2 (Humble)
### Install depending packages
```
pip install --upgrade "empy==3.3.4" lark-parser
pip install catkin_pkg
sudo apt install ros-humble-pcl-ros
```

### Build ROS workspace
```
cd ws_slam
colcon build --symlink-install
```

### Run Fast-LIO2 and Lidar Driver
In terminal one:
```
source install/setup.bash
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml

```
In second terminal
```
source install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

### Stream robot pose via redis
```
python lidar_streamer.py
```