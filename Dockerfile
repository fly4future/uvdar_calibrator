ARG ROS_DISTRO=jazzy
ARG MRS_VERSION=stable_26_10_04
ARG TARGETPLATFORM

ARG PYLON_AMD64_URL=https://downloads-bsl.baslerweb.com/software/pylon%2026.02/pylon-26.02.1_linux-x86_64_debs.tar.gz
ARG PYLON_ARM64_URL=https://downloads-ctf.baslerweb.com/dg51pdwahxgw/6EUPou5YlZl6ZRZ1jxRaju/52f50198a52d4dfe42a35ef60b3dedb3/pylon-26.02.1_linux-aarch64_debs.tar.gz
ARG PYLON_SUPPLEMENTARY_AMD64_URL=https://downloads-ctf.baslerweb.com/dg51pdwahxgw/1SEETUq8OWph5FCGgdtgKV/42f1b4b97db98eaf3b552345cf2c8031/pylon-supplementary-package-for-blaze-1.7.3.73dbe706a_amd64.deb
ARG PYLON_SUPPLEMENTARY_ARM64_URL=https://downloads-ctf.baslerweb.com/dg51pdwahxgw/rS4LtLwhOsISlqjWeMKnY/a0fd95ed967c92bd5d928b77e1955db8/pylon-supplementary-package-for-blaze-1.7.3.73dbe706a_arm64.deb

FROM ctumrs/mrs_uav_system:${MRS_VERSION} AS base

ARG ROS_DISTRO
ARG TARGETPLATFORM
ARG PYLON_AMD64_URL
ARG PYLON_ARM64_URL
ARG PYLON_SUPPLEMENTARY_AMD64_URL
ARG PYLON_SUPPLEMENTARY_ARM64_URL

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TERM=xterm-256color
ENV COLORTERM=truecolor
ENV RCUTILS_COLORIZED_OUTPUT=1
ENV PYTHONDONTWRITEBYTECODE=1

# Download the Basler Pylon SDK (.debs tars) for the target platform. TARGETPLATFORM is set automatically by buildkit
RUN mkdir -p /tmp/basler-camera-drivers && \
    case "${TARGETPLATFORM:-linux/$(dpkg --print-architecture)}" in \
      "linux/amd64") curl -fsSL "${PYLON_AMD64_URL}" | tar -xzf - -C /tmp/basler-camera-drivers && curl -fsSL "${PYLON_SUPPLEMENTARY_AMD64_URL}" --output /tmp/basler-camera-drivers/supplementary.deb ;; \
      "linux/arm64") curl -fsSL "${PYLON_ARM64_URL}" | tar -xzf - -C /tmp/basler-camera-drivers && curl -fsSL "${PYLON_SUPPLEMENTARY_ARM64_URL}" --output /tmp/basler-camera-drivers/supplementary.deb ;; \
      *) echo "Unsupported platform: ${TARGETPLATFORM}" && exit 1 ;; \
    esac && \
    ls /tmp/basler-camera-drivers && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    bash-completion \
    build-essential \
    git \
    tmux \
    tmuxinator \
    python3-pip \
    python3-pytest \
    libcamera-dev \
    ros-${ROS_DISTRO}-ament-cmake-ros \
    ros-${ROS_DISTRO}-libcamera-ros-driver \
    /tmp/basler-camera-drivers/*.deb \
    && rm -rf /tmp/basler-camera-drivers /var/lib/apt/lists/*

# pypylon is used to list Basler cameras by serial number
RUN pip3 install --break-system-packages pypylon

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && (rosdep init || true) && rosdep update

# The Basler driver goes into its own underlay workspace, so building the mounted
# calibrator workspace never rebuilds it
RUN git clone --depth 1 --branch ros2 https://github.com/ctu-mrs/pylon-ros-camera.git /deps_ws/src/pylon-ros-camera

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt-get update && \
    rosdep install --from-paths /deps_ws/src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Clone the libcamera driver at the release the MRS image ships and patch in runtime
# exposure (~/expose_us): the stock package only reads control/exposure_time at
# startup. Cloned after rosdep on purpose -- its mrs_lib dependency comes from the
# installed apt packages (kept as the underlay), not from a rosdep key.
COPY patches/0001-libcamera-expose-us.patch /tmp/libcamera-expose-us.patch
RUN git clone https://github.com/ctu-mrs/libcamera_ros_driver.git /deps_ws/src/libcamera_ros_driver && \
    git -C /deps_ws/src/libcamera_ros_driver checkout 7561354be28813632538d184c8e7d1ed99a1ea8e && \
    git -C /deps_ws/src/libcamera_ros_driver apply /tmp/libcamera-expose-us.patch && \
    rm /tmp/libcamera-expose-us.patch

WORKDIR /deps_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build --cmake-args -DBUILD_TESTING=OFF

# Only package.xml is needed to resolve the calibrator's dependencies
COPY package.xml /tmp/uvdar_calibrator/package.xml

RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    apt-get update && \
    rosdep install --from-paths /tmp/uvdar_calibrator --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/* /tmp/uvdar_calibrator

RUN mkdir -p /ros2_ws/src /calibration_output
WORKDIR /ros2_ws

# Source ROS and the driver underlay, plus the calibrator workspace once it has been built
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /deps_ws/install/setup.bash" >> /root/.bashrc && \
    echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Development image (./docker-dev.sh): the checkout is mounted at
# /ros2_ws/src/uvdar_calibrator and rebuilt on every start (only this package,
# and incrementally, since build/install live in volumes)
FROM base AS dev
CMD ["bash", "-c", "colcon build --symlink-install && cd src/uvdar_calibrator && exec python3 -m uvdar_calibrator.apps.camera_wizard"]

# Production image (default target): the source is baked in and prebuilt
FROM base
ARG ROS_DISTRO
COPY . /ros2_ws/src/uvdar_calibrator
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build --symlink-install
WORKDIR /ros2_ws/src/uvdar_calibrator
CMD ["python3", "-m", "uvdar_calibrator.apps.camera_wizard"]
