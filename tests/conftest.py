from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def demo_yaml_path(project_root: Path) -> Path:
    return project_root / "configs" / "demo.yaml"
