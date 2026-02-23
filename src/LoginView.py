# src/LoginView.py
import flet as ft
from src.CoreApp import CoreApp

class LoginView(ft.View):
    def __init__(self, core: CoreApp, on_login):
        super().__init__(route="/login", horizontal_alignment="center", vertical_alignment="center")
        self.core = core
        self.on_login = on_login

        self.text_reg_title = ft.Text("Регистрация", size=26, weight="bold", color=ft.Colors.BLUE_GREY_800)
        self.text_auth_title = ft.Text("Авторизация", size=26, weight="bold", color=ft.Colors.BLUE_GREY_800)
        self.field_login = ft.TextField(width=300, label="Имя пользователя", on_change=self.validate)
        self.field_password = ft.TextField(width=300, label="Пароль", password=True, on_change=self.validate)
        self.text_error = ft.Text(value="", color='red')
        self.button_register = ft.ElevatedButton(width=200,content='Создать пользователя', on_click=self.handle_register, disabled=True)
        self.button_login = ft.ElevatedButton(width=100, content="Войти", on_click=self.handle_login, disabled=True)
        
        self.view_registry = ft.Column(
            [
                self.text_reg_title,
                self.field_login,
                self.field_password,
                self.text_error,
                self.button_register
            ],
            horizontal_alignment="center"
        )

        self.view_authorization = ft.Column(
            [
                self.text_auth_title,
                self.field_login,
                self.field_password,
                self.text_error,
                self.button_login
            ],
            horizontal_alignment="center"
        )

        self.navigation_bar = ft.NavigationBar(
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.VERIFIED_USER_OUTLINED, label="Вход"),
                ft.NavigationBarDestination(icon=ft.Icons.VERIFIED_USER, label="Регистрация")
            ], 
            on_change=self.navigate
        )

        self.controls = [
            self.navigation_bar,
            self.view_authorization  # по умолчанию
        ]

    def navigate(self, e):
        chosen = self.navigation_bar.selected_index
        self.controls.clear()
        self.field_login.value = ""
        self.field_password.value = ""
        self.text_error.value = ""
        if chosen == 0: 
            self.controls.append(self.navigation_bar)
            self.controls.append(self.view_authorization)
        elif chosen == 1:
            self.controls.append(self.navigation_bar)
            self.controls.append(self.view_registry)
        self.update()

    def validate(self, e):
        self.button_register.disabled = not (
            self.field_login.value and self.field_password.value
        )
        self.button_login.disabled = not (
            self.field_login.value and self.field_password.value
        )
        self.update()

    def handle_register(self, e):
        user_login = self.field_login.value
        user_password = self.field_password.value
        user = self.core.register_user(user_login, user_password)
        if user is None:
            self.text_error.value = "Ошибка: Пользователь с таким именем уже существует"
            self.update()
            return
        self.on_login(user)
        

    def handle_login(self, e):
        user_login = self.field_login.value
        user_password = self.field_password.value
        user = self.core.authenticate_user(user_login, user_password)
        if user is None:
            self.text_error.value = "Неверный логин или пароль"
            self.update()
            return
        self.text_error.value = ""
        self.on_login(user)
