#!/bin/bash
set -ex

# [Pre-setup] Check required env variables.
: "${TOOLKIT_VERSION:?Error: TOOLKIT_VERSION is not set. Please define it in the GitHub Actions workflow.}"
: "${SWIFT_VERSION:?Error: SWIFT_VERSION is not set. Please define it in the GitHub Actions workflow.}"
: "${SWIFT_SIGNING_KEY:?Error: SWIFT_SIGNING_KEY is not set. Please define it in the GitHub Actions workflow.}"

# [Pre-setup] System compatibility checks.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: This script is intended for Linux only."
    echo "Detected OS: $(uname -s)"
    exit 1
fi
export ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    echo "Error: This script is intended for x86_64 arch only."
    echo "Detected arch: $(uname -m)"
    exit 1
fi
ID=$(grep '^ID=' /etc/os-release | cut -d'=' -f2 || true)
VERSION_ID=$(grep '^VERSION_ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"' || true)
if [[ "$ID" != "ubuntu" || "$VERSION_ID" != "24.04" ]]; then
    PRETTY_NAME=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d'=' -f2 | tr -d '"' || true)
    echo "Error: This script is intended for Ubuntu 24.04 only."
    echo "Detected OS: ${PRETTY_NAME:-"Unknown Linux"}"
    exit 1
fi
SWIFT_PLATFORM=ubuntu24.04

# [Setup] Install dependencies.
## Common deps.
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y \
    binutils \
    build-essential \
    cmake \
    curl \
    dkms \
    git \
    gnupg2 \
    libblas-dev \
    libc6-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libgcc-13-dev \
    liblapacke-dev \
    libncurses-dev \
    libopenblas-dev \
    libpython3-dev \
    libsqlite3-0 \
    libstdc++-13-dev \
    libxml2-dev \
    libz3-dev \
    ninja-build \
    pkg-config \
    python3-lldb \
    tzdata \
    ubuntu-drivers-common \
    unzip \
    zip \
    zlib1g-dev

## Install MPI.
sudo apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

## Install CUDA toolkit + NCCL.
CUDA_MAJOR_VERSION=${TOOLKIT_VERSION%.*}
CUDA_TOOLKIT_PKG="cuda-toolkit-${TOOLKIT_VERSION#cuda-}"
CUDNN_PKG="libcudnn9-dev-${CUDA_MAJOR_VERSION}"
CUDA_PACKAGES="$CUDNN_PKG $CUDA_TOOLKIT_PKG"

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$ARCH/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y \
    libnccl2 \
    libnccl-dev \
    $CUDA_PACKAGES
CUDA_BIN_PATH="/usr/local/${TOOLKIT_VERSION}/bin"
export PATH="$CUDA_BIN_PATH:$PATH"
rm cuda-keyring_1.1-1_all.deb

## Sanity checks - system requirements.
echo "NVIDIA Driver Packages Available:"
sudo ubuntu-drivers list --gpgpu
echo "NVIDIA Driver Version:"
cat /proc/driver/nvidia/version || echo "nvidia driver not found"
echo "Installed NVIDIA and CUDA packages:"
dpkg -l | egrep "cuda|nvidia" -i
echo "DKMS Status:"
dkms status || echo "dkms not found"
echo "NVIDIA-SMI Status:"
nvidia-smi || echo "nvidia-smi not found"

## Install Swift toolchain manually (status 12/18/2025: no official swiftlang ubuntu package available yet).
# See for example https://github.com/swiftlang/swift-docker/blob/main/6.2/ubuntu/24.04/Dockerfile
SWIFT_WEBROOT="https://download.swift.org"
case "$ARCH" in
  x86_64)
    OS_ARCH_SUFFIX=""
    ;;
  aarch64|arm64)
    OS_ARCH_SUFFIX="-aarch64"
    ;;
  *)
    echo "Error: Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac
SWIFT_BRANCH=$(echo "$SWIFT_VERSION" | tr '[:upper:]' '[:lower:]')
SWIFT_WEBDIR="$SWIFT_WEBROOT/$SWIFT_BRANCH/$(echo $SWIFT_PLATFORM | tr -d .)$OS_ARCH_SUFFIX"
SWIFT_BIN_URL="$SWIFT_WEBDIR/$SWIFT_VERSION/$SWIFT_VERSION-$SWIFT_PLATFORM$OS_ARCH_SUFFIX.tar.gz"
SWIFT_SIG_URL="$SWIFT_BIN_URL.sig"
GNUPGHOME="$(mktemp -d)"
curl -fsSL "$SWIFT_BIN_URL" -o swift.tar.gz "$SWIFT_SIG_URL" -o swift.tar.gz.sig
gpg --batch --quiet --keyserver keyserver.ubuntu.com --recv-keys "$SWIFT_SIGNING_KEY"
gpg --batch --verify swift.tar.gz.sig swift.tar.gz
# - Unpack the toolchain, set libs permissions, and clean up.
sudo tar -xzf swift.tar.gz --directory / --strip-components=1
sudo chmod -R o+r /usr/lib/swift
rm -rf "$GNUPGHOME" swift.tar.gz.sig swift.tar.gz
swift --version
