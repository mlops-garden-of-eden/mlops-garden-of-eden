import os
import sys
import shutil
import pytest

# Remove the tmp dir, if it is there
if os.path.exists('/tmp/mlops-garden-of-eden'):
    shutil.rmtree('/tmp/mlops-garden-of-eden', ignore_errors=True)

# Copy to /tmp to avoid cache issues
shutil.copytree(
    '/Workspace/Users/chenjoachim@cs.toronto.edu/mlops-garden-of-eden',
    '/tmp/mlops-garden-of-eden',
    dirs_exist_ok=True
)

os.chdir('/tmp/mlops-garden-of-eden')
sys.path.insert(0, '/tmp/mlops-garden-of-eden')

# Run pytest
exit_code = pytest.main(['tests/', '-v', '-p', 'no:cacheprovider'])

if exit_code == 0:
    print("\nAll tests passed!")
else:
    print(f"\nTests failed with exit code: {exit_code}")