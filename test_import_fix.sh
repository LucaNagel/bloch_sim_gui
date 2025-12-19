#!/bin/bash
set -e

echo "=== Setting up test environment ==="
# 1. Create a fresh virtual environment
python3 -m venv venv_conflict_test
source venv_conflict_test/bin/activate

echo "=== Installing PyQt6 (to simulate conflict) ==="
# 2. Install PyQt6 to create the conflict condition
pip install PyQt6 numpy scipy matplotlib h5py imageio

echo "=== Installing blochsimulator from local wheel ==="
# 3. Find and install the wheel we just built (pick the latest one)
WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: Wheel file not found!"
    exit 1
fi
echo "Installing: $WHEEL_FILE"
pip install --force-reinstall "$WHEEL_FILE"

echo "=== Running Import Test ==="
# 4. Attempt to import blochsimulator.
# If the fix works, this should NOT trigger PyQt5 imports and thus NOT crash.
python3 -c "
import sys
try:
    import blochsimulator
    print('SUCCESS: blochsimulator imported without error.')
    
    # Optional: Check that PyQt5 modules are NOT loaded
    if 'PyQt5' in sys.modules:
        print('WARNING: PyQt5 was loaded during import!')
    else:
        print('VERIFIED: PyQt5 was NOT loaded.')
        
except Exception as e:
    print(f'FAILURE: Import failed with error: {e}')
    sys.exit(1)
"

echo "=== Cleaning up ==="
deactivate
rm -rf venv_conflict_test
echo "Test complete."
