FROM ubuntu:18.04
RUN apt update
RUN apt install -y software-properties-common wget

# Install python and the dependency packages
RUN apt install -y python3-pip
RUN python3 -m pip install numpy scipy astropy matplotlib vectormath pytest

# Install the GNU compiler versions
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install -y build-essential
RUN apt install -y gcc-6 g++-6 gcc-7 g++-7 gcc-8 g++-8 gcc-9 g++-9 gcc-10 g++-10

# Install the Clang compiler versions
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
RUN add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main"
RUN add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main"
RUN add-apt-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main"
RUN apt update
RUN apt-get install -y clang-6.0 lld-6.0 clang-7 lld-7 clang-8 lld-8 clang-9 lld-9 clang-10 lld-10

# Setup update-alternatives for cc
RUN update-alternatives --remove-all c++
RUN ln -s -T /usr/bin/clang-10 /usr/bin/clang
RUN ln -s -T /usr/bin/clang++-10 /usr/bin/clang++
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 10 --slave /usr/bin/c++ c++ /usr/bin/g++
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 9 --slave /usr/bin/c++ c++ /usr/bin/clang++

# # Setup update-alternatives gcc
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 6 --slave /usr/bin/g++ g++ /usr/bin/g++-6
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7 --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 --slave /usr/bin/g++ g++ /usr/bin/g++-8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 --slave /usr/bin/g++ g++ /usr/bin/g++-10

# # Setup update-alternatives for clang
RUN ln -s -T /usr/include/c++/10 /usr/include/c++/999
RUN ln -s -T /usr/lib/gcc/x86_64-linux-gnu/10 /usr/lib/gcc/x86_64-linux-gnu/999
RUN ln -s -T /usr/include/x86_64-linux-gnu/c++/10 /usr/include/x86_64-linux-gnu/c++/999

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 6 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 --slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/6 --slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/6 --slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/6
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 7 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-7 --slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/7 --slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/7 --slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/7
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-8 8 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-8 --slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/8 --slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/8 --slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/8
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 9 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-9 --slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/9 --slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/9 --slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/9
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 10 --slave /usr/bin/clang++ clang++ /usr/bin/clang++-10 --slave /usr/lib/gcc/x86_64-linux-gnu/999 clang-gcc-toolchain /usr/lib/gcc/x86_64-linux-gnu/10 --slave /usr/include/c++/999 clang-gcc-toolchain-includes /usr/include/c++/10 --slave /usr/include/x86_64-linux-gnu/c++/999 clang-gcc-toolchain-includes-x86_64 /usr/include/x86_64-linux-gnu/c++/10

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt update
RUN apt-get -y install cmake

# Install boost headers

# Install opencv

# Install Eigen

# Copy the python test script

# Run the python test script

CMD ["/bin/bash"]