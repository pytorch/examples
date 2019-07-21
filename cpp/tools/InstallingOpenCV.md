# Installing OpenCV 

## Linux with Package Manager

### Arch Linux

```shell
pacman -Syu base-devel opencv
```

### Fedora

```shell
sudo dnf install opencv opencv-dev
```

## Linux From Source

Required Packages:

```shell
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

Optional Packages:

```shell
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

Building from Source:

```shell
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8 # runs 8 jobs in parallel
sudo make install
```

## Windows

You can download the pre-built libraries from [OpenCV releases](https://github.com/opencv/opencv/releases) and install them easily.
