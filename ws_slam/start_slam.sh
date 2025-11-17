#!/usr/bin/env bash
set -e

source install/setup.bash


ros2 launch livox_ros_driver2 msg_MID360_launch.py&
PID1=$!

sleep 3
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml &
PID2=$!

sleep 3
python ../lidar_streamer.py &
PID3=$!

wait $PID1 $PID2 $PID3
