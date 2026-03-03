#!/bin/bash
# Setup script for nQPU-Metal JAX integration

set -e  # Exit on error

echo "======================================"
echo "nQPU-Metal JAX Integration Setup"
echo "======================================"
echo ""

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Minimum Python 3.9 required for JAX
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Step 1: Install Python dependencies
echo "Step 1: Installing Python dependencies..."
pip3 install --upgrade pip

# Detect GPU support
if [ "$PLATFORM" == "Darwin" ]; then
    # macOS - use Metal backend
    echo "  Installing JAX with Metal support (Apple Silicon)..."
    pip3 install jax jaxlib
elif command -v nvidia-smi &> /dev/null; then
    # NVIDIA GPU detected
    echo "  Installing JAX with CUDA 12 support..."
    pip3 install "jax[cuda12]"
else
    # CPU only
    echo "  Installing JAX (CPU only)..."
    pip3 install jax jaxlib
fi

# Install other dependencies
pip3 install -r python/requirements_jax.txt

echo "✓ Python dependencies installed"
echo ""

# Step 2: Build Rust bindings
echo "Step 2: Building Rust-Python bindings..."

# Check if maturin is available
if ! command -v maturin &> /dev/null; then
    echo "  Installing maturin..."
    pip3 install maturin
fi

# Build nqpu-metal with Python bindings
echo "  Building nqpu-metal (this may take a few minutes)..."
maturin develop --release --features python

echo "✓ Rust bindings built"
echo ""

# Step 3: Verify installation
echo "Step 3: Verifying installation..."

python3 -c "
import sys
sys.path.insert(0, 'python')

# Test JAX
try:
    import jax
    print(f'✓ JAX {jax.__version__} imported')
    print(f'  Backend: {jax.default_backend()}')
except ImportError as e:
    print(f'✗ JAX import failed: {e}')
    sys.exit(1)

# Test Rust bindings
try:
    from nqpu_metal import PyJAXCircuit
    print('✓ Rust bindings imported')
    circuit = PyJAXCircuit(2)
    circuit.h(0)
    print(f'  Circuit created: {circuit}')
except ImportError as e:
    print(f'✗ Rust bindings import failed: {e}')
    sys.exit(1)

# Test nqpu_jax
try:
    from nqpu_jax import quantum_expectation, check_installation
    status = check_installation()
    print('✓ nqpu_jax imported')
    print(f'  Status: {status}')
    if not all(status.values()):
        print('  WARNING: Some components missing')
except ImportError as e:
    print(f'✗ nqpu_jax import failed: {e}')
    sys.exit(1)

print('')
print('✓ All components verified successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Installation Complete! 🎉"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "  1. Add to PYTHONPATH:"
    echo "     export PYTHONPATH=\"$(pwd)/python:\$PYTHONPATH\""
    echo ""
    echo "  2. Run tests:"
    echo "     python3 python/test_jax_integration.py"
    echo ""
    echo "  3. Try the Jupyter notebook:"
    echo "     jupyter notebook python/jax_integration_demo.ipynb"
    echo ""
    echo "  4. Read the docs:"
    echo "     cat python/README_JAX.md"
    echo ""
else
    echo ""
    echo "======================================"
    echo "Installation Failed"
    echo "======================================"
    echo ""
    echo "Please check the errors above and try:"
    echo "  - Updating Python to 3.9+"
    echo "  - Installing Rust toolchain"
    echo "  - Checking PYTHONPATH"
    echo ""
    exit 1
fi
