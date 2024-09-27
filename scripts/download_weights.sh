#!/bin/bash
MODEL_URL="https://storage.googleapis.com/niantic-lon-static/research/handdgp/handdgp_freihand.ckpt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/weights"
OUTPUT_FILE="$OUTPUT_DIR/handdgp_freihand.ckpt"
mkdir -p "$OUTPUT_DIR"
echo "Downloading the model to $OUTPUT_FILE..."
curl -L -o "$OUTPUT_FILE" "$MODEL_URL"
if [ $? -eq 0 ]; then
    echo "Model downloaded successfully and saved to $OUTPUT_FILE"
else
    echo "Failed to download the model."
    exit 1
fi
