#tests/test_flet_icons.py
import flet as ft

def main(page: ft.Page):
    icons = [icon for icon in dir(ft.Icons)]
    #print(icons)
    print([attr for attr in dir(ft.alignment) if not attr.startswith('_')])

ft.app(target=main)
