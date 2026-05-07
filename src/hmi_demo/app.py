from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from hmi_demo.config import load_config
from hmi_demo.ui.hmi_window import HMIWindow

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"


def main() -> int:
    try:
        cfg = load_config(DEFAULT_CONFIG)
    except (FileNotFoundError, ValueError) as exc:
        app = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "Configuration error",
            f"Failed to load configuration from {DEFAULT_CONFIG}:\n\n{exc}",
        )
        return 1

    app = QApplication(sys.argv)
    try:
        win = HMIWindow(cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        QMessageBox.critical(None, "Startup failed", f"{exc}")
        return 1
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
