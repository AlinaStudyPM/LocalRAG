# src/RAGApp.py
from src.CoreApp import CoreApp
from src.FileUploader import FileUploaderWeb, FileUploaderDesktop, FileUploaderConsole
from src.FletUI import FletUI

class RAGApp:
    """
    Основной класс приложения
    """
    def __init__(self):
        self.core = CoreApp()
        
    def run_web(self):
        self.uploader = FileUploaderWeb(self.core.config)
        self.ui = FletUI(core=self.core, uploader=self.uploader)
        self.ui.run_web()

    def run_desktop(self):
        self.uploader = FileUploaderDesktop(self.core.config)
        self.ui = FletUI(core=self.core, uploader=self.uploader)
        self.ui.run_desktop()
