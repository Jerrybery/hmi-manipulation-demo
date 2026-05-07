from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from hmi_demo.config import load_config
from hmi_demo.ui.hmi_window import HMIWindow

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"


def main() -> int:
    cfg = load_config(DEFAULT_CONFIG)
    app = QApplication(sys.argv)
    win = HMIWindow(cfg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
