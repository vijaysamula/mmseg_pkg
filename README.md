# mmseg_deploy

This repository represents the real-time implementation of 2D object segmentation by use of [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation). The execution is carried out entirely in [docker](https://www.docker.com/).

## Docker 
```
git clone https://github.com/vijaysamula/mmseg_pkg.git
docker build -t mmseg .
```

#### 1) To create container of the image.
```
nvidia-docker run  -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:$HOME/.Xauthority -v /path/to/shared_dir:/shared/ --net=host --pid=host --ipc=host --cap-add=SYS_PTRACE --name mmseg_cont mmseg:latest /bin/bash
```

#### 2) Open the container and follow the below procedure to run.
```
cd mmseg_ws
catkin_make
source devel/setup.bash
```

#### 3) Change the model_path in [mmseg_start_rgb.launch](/mmseg_deploy/launch/mmseg_start_rgb.launch) and [mmseg_start_thermal.launch](/mmseg_deploy/launch/mmseg_start_thermal.launch). Change the bag file path in [play_rosbag.launch](/mmseg_deploy/launch/play_rosbag.launch).

#### 4) Finally run the launch file.
```
roslaunch mmdet_deploy mmdet_start_rgb.launch

roslaunch mmdet_deploy mmdet_start_thermal.launch
```