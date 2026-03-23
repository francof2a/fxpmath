import os
import re
import subprocess
import sys
from pathlib import Path


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _source_version(repo_root: Path) -> str:
    init_file = repo_root / "fxpmath" / "__init__.py"
    content = init_file.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    assert match is not None, "Could not read __version__ from fxpmath/__init__.py"
    return match.group(1)


def test_install_from_source_in_isolated_venv(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    expected_version = _source_version(repo_root)

    venv_dir = tmp_path / "install-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
        text=True,
    )

    venv_python = _venv_python(venv_dir)

    install = subprocess.run(
        [str(venv_python), "-m", "pip", "install", str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert install.returncode == 0, (
        "pip install from source failed\n"
        f"stdout:\n{install.stdout}\n"
        f"stderr:\n{install.stderr}"
    )

    check_import = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "import fxpmath; "
                "from pathlib import Path; "
                "print(fxpmath.__version__); "
                "print(Path(fxpmath.__file__).resolve())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert check_import.returncode == 0, (
        "Import check in isolated venv failed\n"
        f"stdout:\n{check_import.stdout}\n"
        f"stderr:\n{check_import.stderr}"
    )

    lines = [line.strip() for line in check_import.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2, f"Unexpected import check output: {check_import.stdout}"

    installed_version = lines[0]
    imported_from = Path(lines[1])

    assert installed_version == expected_version
    assert str(repo_root) not in str(imported_from)
    assert "site-packages" in str(imported_from).replace("\\", "/")
