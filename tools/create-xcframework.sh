#!/bin/bash

# Build XCFramework script for MLX
# Supports: iOS, iOS Simulator, macOS, Mac Catalyst, tvOS, tvOS Simulator, visionOS, visionOS Simulator
# Produces: MLX.framework, Cmlx.framework, MLXNN.framework, MLXOptimizers.framework

set -e

# Configuration
SCHEME="MLX"
PROJECT="${SCHEME}.xcodeproj"
BUILD_DIR="build"
ARCHIVES_DIR="${BUILD_DIR}/archives"
OUTPUT_DIR="${BUILD_DIR}/output"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Frameworks to package
FRAMEWORKS=("MLX" "Cmlx" "MLXNN" "MLXOptimizers")

# Destinations to build for (ordered array)
PLATFORM_NAMES=("iOS" "macOS" "Mac Catalyst" "tvOS" "visionOS")
PLATFORM_DESTINATIONS=(
    "generic/platform=iOS"
    "generic/platform=macOS"
    "generic/platform=macOS,variant=Mac Catalyst"
    "generic/platform=tvOS"
    "generic/platform=visionOS"
)

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create build directories
setup_directories() {
    log_info "Setting up directories..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${ARCHIVES_DIR}"
    mkdir -p "${OUTPUT_DIR}"
}

# Archive for a specific platform
archive_for_platform() {
    local platform=$1
    local destination=$2
    local archive_path="${ARCHIVES_DIR}/${SCHEME}-${platform}"

    log_info "Archiving for ${platform}..."

    xcodebuild archive \
        -project "${PROJECT}" \
        -scheme "${SCHEME}" \
        -destination "${destination}" \
        -archivePath "${archive_path}" \
        SKIP_INSTALL=NO \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        -quiet

    if [ $? -eq 0 ]; then
        log_info "✓ Archive created: ${archive_path}"
        echo "${archive_path}"
    else
        log_error "Failed to archive for ${platform}"
        exit 1
    fi
}

# Extract framework path from archive
get_framework_path() {
    local archive_path=$1
    local framework_name=$2
    echo "${archive_path}.xcarchive/Products/Library/Frameworks/${framework_name}.framework"
}

# Extract debug symbols path from archive
get_debug_symbols_path() {
    local archive_path=$1
    echo "${archive_path}.xcarchive/Products/Library/Frameworks"
}

# Create XCFramework for a specific framework
create_xcframework_for_framework() {
    local framework_name=$1
    local output_path="${OUTPUT_DIR}/${framework_name}.xcframework"

    log_info "Creating XCFramework for ${framework_name}..."

    # Build the xcodebuild command with all platform frameworks
    local cmd="xcodebuild -create-xcframework"

    for platform in "${!DESTINATIONS[@]}"; do
        local archive_path="${ARCHIVES_DIR}/${SCHEME}-${platform}"
        local framework_path=$(get_framework_path "${archive_path}" "${framework_name}")
        local debug_symbols_path="${archive_path}.xcarchive/Products/Library/Frameworks"

        if [ -d "${framework_path}" ]; then
            cmd+=" -framework ${framework_path}"
            cmd+=" -debug-symbols ${debug_symbols_path}"
        else
            log_warn "Framework not found: ${framework_path}"
        fi
    done

    cmd+=" -output ${output_path}"

    # Execute the command
    eval "${cmd}"

    if [ $? -eq 0 ]; then
        log_info "✓ XCFramework created: ${output_path}"
    else
        log_error "Failed to create XCFramework for ${framework_name}"
        exit 1
    fi
}

# Main build process
main() {
    log_info "Starting XCFramework build for ${SCHEME}"
    echo ""

    setup_directories
    echo ""

    # Step 1: Archive for all platforms
    log_info "Step 1: Archiving for all platforms..."
    echo ""

    for i in "${!PLATFORM_NAMES[@]}"; do
        platform="${PLATFORM_NAMES[$i]}"
        destination="${PLATFORM_DESTINATIONS[$i]}"
        archive_for_platform "${platform}" "${destination}"
    done
    echo ""

    # Step 2: Create XCFrameworks for each framework
    log_info "Step 2: Creating XCFrameworks..."
    echo ""

    for framework in "${FRAMEWORKS[@]}"; do
        create_xcframework_for_framework "${framework}"
    done
    echo ""

    # Summary
    log_info "Build complete!"
    echo ""
    log_info "XCFrameworks created in: ${OUTPUT_DIR}"
    echo ""
    for framework in "${FRAMEWORKS[@]}"; do
        echo "  • ${framework}.xcframework"
    done
    echo ""
    log_info "Archive files stored in: ${ARCHIVES_DIR}"
    echo ""
}

main "$@"
