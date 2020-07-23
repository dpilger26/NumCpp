FROM ubuntu:18.04
RUN \
	# Install the basics
	apt update \
	&& apt install -y software-properties-common wget apt-utils vim git \
	# Add the repos
	&& wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - \
	&& wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& add-apt-repository ppa:ubuntu-toolchain-r/test \
	&& add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main" \
	&& add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main" \
	&& add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main" \
	&& apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
	&& apt update \
	# Install python and the dependency packages
	&& apt install -y python3.7 \
	&& update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 10 \
	&& apt install -y python3-pip \
	&& python3 -m pip install numpy scipy astropy matplotlib vectormath pytest \
	# Install the GNU compiler versions
	&& apt install -y build-essential \
	&& apt install -y gcc-6 g++-6 gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10 \
	# Install the Clang compiler versions
	&& apt install -y clang-6.0 lld-6.0 clang-7 lld-7 clang-8 lld-8 clang-9 lld-9 clang-10 lld-10 \
	# Setup update-alternatives for cc
	&& update-alternatives --remove-all c++ \
	&& ln -s -T /usr/bin/clang-10 /usr/bin/clang \
	&& ln -s -T /usr/bin/clang++-10 /usr/bin/clang++ \
	&& update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 10 --slave /usr/bin/c++ c++ /usr/bin/g++ \
	&& update-alternatives --install /usr/bin/cc cc /usr/bin/clang 9 --slave /usr/bin/c++ c++ /usr/bin/clang++ \
	# # Setup update-alternatives gcc
	&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 6 --slave /usr/bin/g++ g++ /usr/bin/g++-6 \
	&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7 --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
	&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
	&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
	&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 --slave /usr/bin/g++ g++ /usr/bin/g++-10 \
	# Setup update-alternatives for clang
	&& ln -s -T /usr/include/c++/10 /usr/include/c++/999 \
	&& ln -s -T /usr/lib/gcc/x86_64-linux-gnu/10 /usr/lib/gcc/x86_64-linux-gnu/999 \
	&& ln -s -T /usr/include/x86_64-linux-gnu/c++/10 /usr/include/x86_64-linux-gnu/c++/999 \
	# link the GNU 7 headers to Clang 6 since the GNU 6 headers seem to have some issues...
	&& update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 6 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 \
	--slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/7 \
	--slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/7 \
	--slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/7 \
	&& update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 7 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-7 \
	--slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/7 \
	--slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/7 \
	--slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/7 \
	&& update-alternatives --install /usr/bin/clang clang /usr/bin/clang-8 8 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-8 \
	--slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/8 \
	--slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/8 \
	--slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/8 \
	&& update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 9 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-9 \
	--slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/9 \
	--slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/9 \
	--slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/9 \
	&& update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 10 \
	--slave /usr/bin/clang++ clang++ /usr/bin/clang++-10 \
	--slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/10 \
	--slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/10 \
	--slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/10 \
	# Install cmake
	&& apt-get -y install cmake \
	# Build and install boost
	&& wget https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz \
	&& tar -zxvf boost_1_72_0.tar.gz \
	&& rm boost_1_72_0.tar.gz \
	&& cd /boost_1_72_0 \
	&& ./bootstrap.sh \
	&& cp /boost_1_72_0/tools/build/example/user-config.jam /boost_1_72_0/tools/build/src \
	&& echo "using python : 3.7 : /usr/bin/python3 : /usr/include/python3.6m : /usr/lib ;" >> /boost_1_72_0/tools/build/src/user-config.jam \
	&& ./b2 install cxxflags=-fPIC -j8 --with-python --with-filesystem --with-system \
	&& cd / \
	&& rm -rf /boost_1_72_0 \
	# Install opencv
	&& apt install -y pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
	libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev \
	libtbb2 libtbb-dev libdc1394-22-dev \
	&& rm /usr/bin/python \
	&& rm /usr/bin/python2.7 \
	&& mkdir /opencv_build \
	&& cd /opencv_build \
	&& git clone https://github.com/opencv/opencv.git \
	&& git clone https://github.com/opencv/opencv_contrib.git \
	&& cd opencv \
	&& mkdir build \
	&& cd build \
	&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_build/opencv_contrib/modules \
	-D PYTHON_INCLUDE_DIRS=/usr/include/python3.6m \
	-D PYTHON_LIBRARIES=/usr/lib .. \
	&& make -j8 \
	&& make install \
	&& cd / \
	&& rm -rf /opencv_build \
	# Install Eigen
	&& mkdir /eigen \
	&& cd eigen \
	&& wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz \
	&& tar -zxvf eigen-3.3.7.tar.gz \
	&& cd eigen-3.3.7 \
	&& mkdir build \
	&& cd build \
	&& cmake .. \
	&& cmake --build . --config Release --target install \
	&& cd / \
	&& rm -rf /eigen \
	# Make directory to mount to from the host
	&& mkdir /NumCpp
WORKDIR /NumCpp
CMD ["/bin/bash"]
