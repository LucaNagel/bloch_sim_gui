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
                background-color: rgba(30, 34, 42, 245);
                border: 2px solid #149071;
                border-radius: 10px;
                color: white;
            }
            QLabel {
                color: white;
                background: transparent;
            }
            QLabel#step_label {
                font-weight: bold;
                font-size: 13px;
                color: #98F1DB;
            }
            QLabel#instruction_label {
                font-weight: bold;
                font-size: 15px;
                color: white;
            }
            QLabel#description_label {
                font-size: 13px;
                color: #CCCCCC;
                font-style: italic;
            }
            QPushButton {
                background-color: #149071;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1bb38d;
            }
            QPushButton:pressed {
                background-color: #0e6b54;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
            QPushButton#stop_btn {
                background-color: #D32F2F;
                margin-top: 5px;
            }
            QPushButton#stop_btn:hover {
                background-color: #E53935;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Header / Instruction
        self.step_label = QLabel("Step 1/10")
        self.step_label.setObjectName("step_label")
        self.step_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.step_label)

        self.instruction_label = QLabel("Click the button...")
        self.instruction_label.setObjectName("instruction_label")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)

        self.description_label = QLabel("")
        self.description_label.setObjectName("description_label")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setVisible(False)
        layout.addWidget(self.description_label)

        # Controls
        btn_layout = QHBoxLayout()

        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        btn_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_clicked.emit)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)

        self.stop_btn = QPushButton("Stop Tutorial")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)
        self.setFixedWidth(280)

    def update_step(self, current, total, instruction, description=None):
        self.step_label.setText(f"STEP {current + 1} OF {total}")
        self.instruction_label.setText(instruction)

        if description:
            self.description_label.setText(description)
            self.description_label.setVisible(True)
        else:
            self.description_label.setVisible(False)

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
