#!/bin/bash
# Deploy NQPU Drug Design to HuggingFace Spaces
#
# Usage:
#   ./deploy_hf.sh <space-name> [hf-username]
#
# Prerequisites:
#   - huggingface-cli installed and logged in
#   - Git LFS installed

set -e

SPACE_NAME="${1:-nqpu-drug-design}"
HF_USER="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$HF_USER" ]; then
    # Try to get username from huggingface-cli
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1 || echo "")
    if [ -z "$HF_USER" ]; then
        echo "Error: Could not determine HuggingFace username."
        echo "Please login with: huggingface-cli login"
        echo "Or provide username: $0 <space-name> <username>"
        exit 1
    fi
fi

SPACE_ID="${HF_USER}/${SPACE_NAME}"
TEMP_DIR=$(mktemp -d)

echo "============================================"
echo "NQPU Drug Design - HuggingFace Spaces Deploy"
echo "============================================"
echo ""
echo "Space: $SPACE_ID"
echo "Temp dir: $TEMP_DIR"
echo ""

# Check if space exists
if huggingface-cli repo info "$SPACE_ID" &>/dev/null; then
    echo "Space exists, cloning..."
    cd "$TEMP_DIR"
    git clone "https://huggingface.co/spaces/$SPACE_ID" .
else
    echo "Creating new space..."
    huggingface-cli repo create "$SPACE_NAME" --type space --sdk gradio
    cd "$TEMP_DIR"
    git clone "https://huggingface.co/spaces/$SPACE_ID" .
fi

# Copy files
echo "Copying files..."
cp "$SCRIPT_DIR/app.py" .
cp "$SCRIPT_DIR/nqpu_drug_design.py" .
cp "$SCRIPT_DIR/requirements.txt" .
cp "$SCRIPT_DIR/README_HF.md" README.md

# Create .gitattributes for LFS
echo "*.bin filter=lfs diff=lfs merge=lfs -text" > .gitattributes

# Commit and push
echo ""
echo "Committing changes..."
git add .
git config user.email "deploy@nqpu.local" 2>/dev/null || true
git config user.name "NQPU Deploy" 2>/dev/null || true
git commit -m "Update NQPU Drug Design Platform" || echo "Nothing to commit"

echo "Pushing to HuggingFace..."
git push

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "============================================"
echo "Deployment complete!"
echo "============================================"
echo ""
echo "Your space is available at:"
echo "  https://huggingface.co/spaces/$SPACE_ID"
echo ""
