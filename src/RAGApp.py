# src/RAGApp.py
from CoreApp import CoreApp
from FletUI import FletUI

class RAGApp:
    """
    Основной класс приложения
    """
    def __init__(self):
        self.core = CoreApp()
        self.ui = FletUI(core=self.core)

    def run(self):
        self.ui.run()