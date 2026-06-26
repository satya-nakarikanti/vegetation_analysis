"""Verify that the Phase 1 development environment is ready."""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyCheck:
    """Result of checking whether a Python package can be imported."""

    package_name: str
    import_name: str
    installed: bool


def check_dependency(
    package_name: str,
    import_name: str | None = None,
) -> DependencyCheck:
    """Check whether a dependency is importable in the active Python environment."""

    module_name = import_name or package_name
    return DependencyCheck(
        package_name=package_name,
        import_name=module_name,
        installed=importlib.util.find_spec(module_name) is not None,
    )


def main() -> int:
    """Run all environment checks and return a process exit code."""

    print("Vegetation Analysis - Environment Verification")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    if sys.version_info < (3, 11):  # noqa: UP036
        print("FAIL: Python 3.11 or newer is required.")
        return 1

    dependencies = [
        check_dependency("numpy"),
        check_dependency("opencv-python", "cv2"),
        check_dependency("pillow", "PIL"),
        check_dependency("pydantic"),
        check_dependency("pydantic-settings", "pydantic_settings"),
        check_dependency("python-dotenv", "dotenv"),
        check_dependency("torch"),
        check_dependency("ultralytics"),
    ]

    missing = [item for item in dependencies if not item.installed]

    for item in dependencies:
        status = "OK" if item.installed else "MISSING"
        print(f"{status}: {item.package_name} ({item.import_name})")

    print()
    if missing:
        print(
            "FAIL: Install missing dependencies with: "
            "pip install -r requirements.txt"
        )
        return 1

    print("OK: Environment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
