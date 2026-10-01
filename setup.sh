#!/usr/bin/env bash
# One-click setup on macOS and Linux:  bash setup.sh
# It finds Python 3.12 and runs install.py, which builds venv/, installs the right PyTorch build
# (NVIDIA GPU or CPU), installs the project and checks that everything works.
# Options are passed through:  bash setup.sh --dev   bash setup.sh --cpu   bash setup.sh --fresh
set -euo pipefail
cd "$(dirname "$0")"

# The first command that really is Python 3.12. Checking the version by running it also skips
# shortcuts that exist but do not work, such as the Microsoft Store alias in Git Bash on Windows.
python=""
for candidate in python3.12 python3 python; do
    if "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1; then
        python="$candidate"
        break
    fi
done
if [ -z "$python" ]; then
    echo "Python 3.12 was not found. Install 64-bit Python 3.12"
    echo "(https://www.python.org/downloads/) and run this again."
    exit 1
fi

"$python" install.py "$@"

echo
echo "Setup finished. To play:"
echo "    venv/bin/python src/play_gesture.py"
