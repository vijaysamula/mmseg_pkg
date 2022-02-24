ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5+PTX "
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmsegmentation
RUN conda clean --all

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
ENV HOME /home/wms
RUN git clone  https://github.com/open-mmlab/mmsegmentation.git $HOME/mmsegmentation
WORKDIR $HOME
RUN cd $HOME/mmsegmentation && \
    pip install -r requirements.txt && \
    pip install --no-cache-dir -e .

RUN apt-get update && apt-get install -yqq  build-essential ninja-build \
  python3-dev python3-pip tig apt-utils curl git cmake unzip autoconf autogen \
  libtool mlocate zlib1g-dev python python3-numpy python3-wheel wget \
  software-properties-common openjdk-8-jdk libpng-dev  \
  libxft-dev vim meld sudo ffmpeg python3-pip libboost-all-dev \
  libyaml-cpp-dev -y && updatedb

# Update Cmake
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt update -y && \
    apt install cmake --upgrade -y

# installing libtorch for c++ shared libraries
RUN wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.0%2Bcu111.zip && \
    cd $HOME && mkdir libtorch && unzip libtorch-cxx11-abi-shared-with-deps-1.8.0+cu111.zip -d /opt/libtorch && \
    rm libtorch-cxx11-abi-shared-with-deps-1.8.0+cu111.zip

# install ros
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
  apt update && \ 
  DEBIAN_FRONTEND=noninteractive apt install -yqq ros-melodic-desktop-full && \
  apt-get install python-rosdep && \
  rosdep init && \
  rosdep update

# install catkin tools
RUN apt-get install -y python-pip python-empy 
RUN pip install -U pip catkin-tools trollius

# recommended from nvidia to use the cuda devices
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# clean the cache
RUN apt update && \
  apt autoremove --purge -y && \
  apt clean -y

RUN rm -rf /var/lib/apt/lists/*

# to use nvidia driver from within
LABEL com.nvidia.volumes.needed="nvidia_driver"

# this is to be able to use graphics from the container
# Replace 1000 with your user / group id (if needed)
RUN export uid=1000 gid=1000 && \
  mkdir -p /home/developer && \
  mkdir -p /etc/sudoers.d && \
  echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
  echo "developer:x:${uid}:" >> /etc/group && \
  echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
  chmod 0440 /etc/sudoers.d/developer && \
  chown ${uid}:${gid} -R /home/developer && \
  adduser developer sudo

RUN echo 'source /opt/ros/melodic/setup.bash' >> $HOME/.bashrc && \
  echo 'export PYTHONPATH=/usr/local/lib/python3.5/dist-packages/cv2/:$PYTHONPATH' >> $HOME/.bashrc && \
  echo 'export NO_AT_BRIDGE=1' >> $HOME/.bashrc

RUN mkdir -p mmseg_ws/src && \
    cd mmseg_ws/src 
ADD . $HOME/mmseg_ws/src