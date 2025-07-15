#!/usr/bin/env bash
# setup.sh - environment setup for genai-dd with PyTorch fix and NumPy < 2 constraint

set -e

echo "🛠️  Setting up environment..."

# 1. Check for Python 3
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 is required but not found." >&2
  exit 1
fi

# 2. Create and activate virtual environment
python3 -m venv .venv
echo "✅ Created virtual environment in .venv/"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "✅ Activated virtual environment."

# 3. Upgrade pip
pip install --upgrade pip

# 4. Temporarily remove torch from requirements.txt
grep -v '^torch' requirements/requirements.txt > /tmp/requirements_no_torch.txt

# 5. Install other requirements
pip install -r /tmp/requirements_no_torch.txt

# 6. Pin NumPy to a compatible version
pip install "numpy<2"

# 7. Detect architecture and install appropriate torch build
ARCH=$(uname -m)
OS=$(uname -s)
echo "Detected architecture: $ARCH, OS: $OS"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  echo "🧠 Apple Silicon detected — installing PyTorch with MPS support..."
  pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
else
  echo "🧠 Installing standard PyTorch CPU build..."
  pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
fi

# 8. Done
echo -e "\n✅ Setup complete. To activate your environment:"
echo "   source .venv/bin/activate"

