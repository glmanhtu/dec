FROM nvidia/cuda:8.0-devel-ubuntu14.04

# Install some dep packages

ENV OPENCV_VERSION 2.4.13

RUN apt-get update && \
    apt-get install -y cmake git wget build-essential unzip python2.7 python2.7-dev python-numpy python-scipy && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install OpenCV

RUN cd /usr/local/src && \
    wget -O opencv-$OPENCV_VERSION.zip https://github.com/Itseez/opencv/archive/$OPENCV_VERSION.zip && \
    unzip opencv-$OPENCV_VERSION.zip && \
    cd opencv-$OPENCV_VERSION && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr \
          -D BUILD_opencv_python=on \
          -D BUILD_EXAMPLES=OFF \
          -D CUDA_GENERATION=Auto \
          -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_VTK=ON -D WITH_OPENGL=OFF -D WITH_QT=OFF .. && \
    make -j$(nproc) && \
    make install && \
    cp lib/cv2.so /usr/local/lib/python2.7/dist-packages/ && \
    rm -rf /usr/local/src/opencv-$OPENCV_VERSION.zip /usr/local/src/opencv-$OPENCV_VERSION

# Install some dep packages

ENV CAFFE_PACKAGES libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev python-dev \
  python-pip python-scipy python-matplotlib python-scikits-learn ipython python-h5py python-leveldb python-networkx python-nose python-pandas \
  python-dateutil python-protobuf python-yaml python-gflags python-skimage python-sympy cython \
  libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev libboost1.54-all-dev libatlas-base-dev

RUN apt-get update && \
    apt-get install -y software-properties-common python-software-properties git wget build-essential pkg-config bc unzip cmake && \
    add-apt-repository ppa:boost-latest/ppa && \
    apt-get install -y $CAFFE_PACKAGES && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install -U leveldb  # fix GH Issue #7

# Copy the source files over and build the project

COPY . /usr/local/src/dec
WORKDIR /usr/local/src/dec

RUN cd /usr/local/src/dec/caffe && \
    cp Makefile.config.example Makefile.config && \
    make -j"$(nproc)" all pycaffe