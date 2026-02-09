from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal


class TutorialOverlay(QFrame):
    """
    Floating overlay widget for tutorial controls (Next/Prev/Stop).
    """

    next_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.init_ui()
        self.drag_pos = None

    def init_ui(self):
        # styling
        self.setStyleSheet(
            """
            QFrame {
                background-color: rgba(40, 44, 52, 230);
                border: 1px solid #555;
                border-radius: 8px;
                color: white;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 14px;
                background: transparent;
            }
            QPushButton {
                background-color: #0078D7;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084E3;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton#stop_btn {
                background-color: #D32F2F;
            }
            QPushButton#stop_btn:hover {
                background-color: #E53935;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Header / Instruction
        self.step_label = QLabel("Step 1/10")
        self.step_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.step_label)

        self.instruction_label = QLabel("Click the button...")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)

        # Controls
        btn_layout = QHBoxLayout()

        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        btn_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_clicked.emit)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)

        self.stop_btn = QPushButton("Stop Tutorial")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)
        self.setFixedWidth(250)

    def update_step(self, current, total, instruction):
        self.step_label.setText(f"Step {current + 1}/{total}")
        self.instruction_label.setText(instruction)

        self.prev_btn.setEnabled(current > 0)
        self.next_btn.setText("Finish" if current == total - 1 else "Skip / Next")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drag_pos:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()
