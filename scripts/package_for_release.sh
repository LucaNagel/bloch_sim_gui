#!/bin/bash
set -e

# Define names
APP_NAME="BlochSimulator"
DIST_DIR="dist"

# Get version from git tag, fallback to short hash if no tags
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
ZIP_NAME="${APP_NAME}_macOS_arm64_${VERSION}.zip"

echo "Packaging $APP_NAME for release ($VERSION)..."

if [ ! -d "$DIST_DIR/$APP_NAME.app" ]; then
    echo "Error: $DIST_DIR/$APP_NAME.app not found. Please build it first using scripts/build_pyinstaller.sh."
    exit 1
fi

# Navigate to dist to zip relative to the folder
cd "$DIST_DIR"

# Remove old zips if exist to avoid confusion
rm -f "${APP_NAME}_macOS_arm64_"*.zip

# Zip the app
# -r: recursive
# -y: store symbolic links as the link instead of the referenced file (CRITICAL for macOS apps)
# -q: quiet mode
zip -r -y -q "$ZIP_NAME" "$APP_NAME.app"

echo "✅ Created release artifact: $DIST_DIR/$ZIP_NAME"
echo "   Size: $(du -h "$ZIP_NAME" | cut -f1)"
