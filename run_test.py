import os
import sys
import shutil
import pytest
from pathlib import Path
import tempfile

# Make the script portable: copy the current repository root into a temporary
# directory for isolation. Prefer system temp dir (usually writable) and
# gracefully fall back to a directory under the user's home if creation fails.
try:
    repo_root = Path(__file__).resolve().parent
except NameError:
    repo_root = Path.cwd().resolve()

# Create a writable temp directory
try:
    tmp_dir = Path(tempfile.mkdtemp(prefix="mlops-garden-of-eden-"))
except Exception:
    # Fallback to a directory under the user's home
    home = Path.home()
    tmp_dir = home / "mlops-garden-of-eden-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

# Copy repository into the temp directory
try:
    shutil.copytree(str(repo_root), str(tmp_dir / repo_root.name), dirs_exist_ok=True)
    target_dir = tmp_dir / repo_root.name
except PermissionError as e:
    raise PermissionError(f"Failed to copy repository to temp directory {tmp_dir}: {e}")

os.chdir(str(target_dir))
sys.path.insert(0, str(target_dir))

# Run pytest in the copied tree
exit_code = pytest.main(['tests/', '-v', '-p', 'no:cacheprovider'])

if exit_code == 0:
    print("\nAll tests passed!")
else:
    print(f"\nTests failed with exit code: {exit_code}")