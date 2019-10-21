# lane_detection
ROS bases Lane detection code for the Autonomous Vehicle project. This repository is inspired by https://github.com/FrenkT/LaneTracking. The sample video can be downloaded from https://drive.google.com/open?id=15pWD7tbbDhYkAhP9FU1Wm_rA0oz-E9Xw. Put it in the lane_detection directory

 ## Installation
 ```
 cd catkin_make/src
 git clone https://github.com/sarath-menon/lane_detection/new/master?readme=1
 cd ..
 cd catkin_make
```

 ## Run lane detection on sample video in ROS

 ```
 rosrun lane_detection main.py
```

 ## Using without ROS

In case ROS is not installed in your system or you want to use it in Mac os Windows, then go inside the scripts directory and run the following command.

```
python main_without_ros.py
```
