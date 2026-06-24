#!/bin/bash
# Setup script for trn1.2xlarge Spot instance
# Runs prerequisites.sh logic (DLAMI skips Neuron install) + Python build deps
set -e

echo "=== trn1 Setup Script ==="

. /etc/os-release
export PATH=/opt/aws/neuron/bin:$PATH

# --- Neuron driver (skip if DLAMI already has it) ---
if ! command -v neuron-ls &> /dev/null; then
    echo "=== Installing Neuron driver/runtime/tools ==="
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --dearmor -o /usr/share/keyrings/neuron.gpg
    sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb [signed-by=/usr/share/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
    sudo apt-get update -y
    sudo apt-get install -y linux-headers-$(uname -r) git
    sudo apt-get install -y --upgrade aws-neuronx-dkms aws-neuronx-collectives aws-neuronx-runtime-lib aws-neuronx-tools
else
    echo "✓ Neuron already installed: $(neuron-ls 2>/dev/null | head -2 | tail -1)"
fi

if ! grep -q '/opt/aws/neuron/bin' ~/.bashrc; then
    echo 'export PATH=/opt/aws/neuron/bin:$PATH' >> ~/.bashrc
fi

# --- System build deps ---
echo "=== Installing system dependencies ==="
sudo apt-get update -y
sudo apt-get install -y build-essential patchelf git cmake libgtest-dev libgmock-dev

# --- Bazelisk ---
if ! command -v bazel &> /dev/null; then
    echo "=== Installing Bazelisk ==="
    wget -qO /tmp/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x /tmp/bazelisk
    sudo mv /tmp/bazelisk /usr/local/bin/bazel
fi
echo "Bazel: $(bazel --version 2>/dev/null | head -1)"
echo "patchelf: $(patchelf --version)"

# --- Python venv ---
echo "=== Setting up Python venv ==="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip setuptools wheel

# --- Neuron Python packages ---
echo "=== Installing Neuron Python packages ==="
pip install 'neuronx-cc==2.*' --extra-index-url https://pip.repos.neuron.amazonaws.com
pip install torch-neuronx torchvision --extra-index-url https://pip.repos.neuron.amazonaws.com

# --- Build dependencies for torch-neuronx source ---
echo "=== Installing Python build dependencies ==="
pip install grpcio-tools protobuf

# --- Clone torch-neuronx ---
echo "=== Cloning torch-neuronx ==="
if [ ! -d ~/torch-neuronx ]; then
    gh auth status || { echo "ERROR: Run 'gh auth login --web' first"; exit 1; }
    gh repo clone aws-neuron/torch-neuronx ~/torch-neuronx
fi

# --- Jupyter ---
pip install jupyterlab ipykernel
python -m ipykernel install --user --name neuron --display-name "PyTorch (Neuron)"

echo ""
echo "=== Setup Complete ==="
echo "Activate:    source ~/venv/bin/activate"
echo "Build:       cd ~/torch-neuronx && ./tools/build"
echo "Jupyter:     jupyter lab --no-browser --port=8888"
echo "SSH tunnel:  ssh -i ~/.ssh/neuron-book.pem -L 8888:localhost:8888 ubuntu@18.223.110.221"
