import flet as ft
from src.LoginView import LoginView
from src.MainView import MainView

class FletUI:
    def __init__(self, core):
        self.page: ft.Page | None = None
        self.core = core
        self.file_picker = ft.FilePicker()
        self.file_picker.on_result = self.on_file_picked
        # self.upload_file = FileUploadView()

    def on_file_picked(self, e: ft.FilePickerUploadEvent):
        """Общий обработчик для всех загрузок файлов."""
        if hasattr(self.page, "active_view") and hasattr(self.page.active_view, "on_file_picked"):
            self.page.active_view.on_file_picked(e)


    def show_login(self):
        if self.page is None:
            return
        
        self.page.views.clear()
        if not self.page.web:
            self.page.window.width = 400
            self.page.window.height = 400
            self.page.window.resizable = False
            self.page.window.center()

        login_view = LoginView(
            core=self.core, 
            on_login=self.show_main
        )
        self.page.views.append(login_view)
        self.page.active_view = login_view
        self.page.update()
    
    def show_main(self, user):
        print("Открываем главное окно")
        if self.page is None:
            return
        
        self.page.views.clear()
        if self.file_picker not in self.page.overlay:
            self.page.overlay.append(self.file_picker)

        if not self.page.web:
            self.page.window.width = 1300
            self.page.window.height = 700
            self.page.window.resizable = False
            self.page.window.center()

        main_view = MainView(
            on_logout=self.show_login
        )
        self.page.views.append(main_view)
        # self.page.views.append(self.upload_file)
        self.page.update()
        self.page.active_view = main_view
        main_view.start(core=self.core, current_user=user)

    
    def main(self, page: ft.Page):
        self.page = page
        page.theme_mode = "light"
        if not self.page.web:
            page.window_frameless = True
        self.show_login()

    def run(self):
        ft.app(
            target=self.main,
            view=ft.AppView.WEB_BROWSER,
            port=8550,           
            host="0.0.0.0"       # важно для Docker!
        )
