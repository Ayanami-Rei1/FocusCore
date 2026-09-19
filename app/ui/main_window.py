from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QListWidget, QStackedWidget, QLabel
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusCore")
        self.resize(800, 600)
        self._built_menu()
        self._built_pages()
        self._built_layout()

    def _build_menu(self):
        self.menu = QListWidget()
        self.menu.addItems([
            "Источник видео",
            "Параметры анализа",
            "Лекция",
            "Отчёты",
            "История",
            "Приватность",
            "Помощь",
        ])
        self.menu.setFixedWidth(200)

    def _built_pages(self):
        self.stack = QStackedWidget()
        for name in ["Источник", "Параметры", "Лекция",
            "Отчёты", "История", "Приватность", "Помощь"]:
            self.stack.addWidget(QLabel(f"Страница: {name}"))

    def _built_layout(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(self.menu)
        layout.addWidget(self.stack)
        self.setCentralWidget(container)
        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.menu.setCurrentRow(0)



