from typing import Callable
import tkinter as tk
from tkinter import ttk
from os.path import join as joinpath


def _resource(*path: str) -> str:
    return joinpath("resources", *path)


class ControlsWindow(tk.Tk):

    WINDOW_TITLE = "Field Line Simulator - Controls"

    SAVE_BTN_IMG = joinpath("icons", "save.png")
    LOAD_BTN_IMG = joinpath("icons", "load.png")
    ADD_BTN_IMG = joinpath("icons", "placeholder.png")
    DELETE_BTN_IMG = joinpath("icons", "placeholder.png")
    SETTINGS_BTN_IMG = joinpath("icons", "placeholder.png")
    HELP_BTN_IMG = joinpath("icons", "placeholder.png")
    RECALCULATE_BTN_IMG = joinpath("icons", "placeholder.png")

    def __init__(self,
                 save_callback: Callable[[], None],
                 load_callback: Callable[[], None],
                 add_callback: Callable[[], None],
                 delete_callback: Callable[[], None],
                 settings_callback: Callable[[], None],
                 help_callback: Callable[[], None],
                 recalculate_callback: Callable[[], None]):

        super().__init__()

        # Setup

        self.title(ControlsWindow.WINDOW_TITLE)

        self.button_frame = tk.Frame(self)

        # Load button images
        # They are stored in member variables so they aren't garbage collected

        self.btn_img_save = tk.PhotoImage(file=_resource(ControlsWindow.SAVE_BTN_IMG))
        self.btn_img_load = tk.PhotoImage(file=_resource(ControlsWindow.LOAD_BTN_IMG))
        self.btn_img_add = tk.PhotoImage(file=_resource(ControlsWindow.ADD_BTN_IMG))
        self.btn_img_delete = tk.PhotoImage(file=_resource(ControlsWindow.DELETE_BTN_IMG))
        self.btn_img_settings = tk.PhotoImage(file=_resource(ControlsWindow.SETTINGS_BTN_IMG))
        self.btn_img_help = tk.PhotoImage(file=_resource(ControlsWindow.HELP_BTN_IMG))
        self.btn_img_recalculate = tk.PhotoImage(file=_resource(ControlsWindow.RECALCULATE_BTN_IMG))

        # Create buttons

        self.__place_button(self.__create_button("Save (Ctrl+S)", save_callback, self.btn_img_save),
                            0, 0
        )
        self.__place_button(self.__create_button("Open (Ctrl+O)", load_callback, self.btn_img_load),
                            0, 1
        )
        self.__place_button(self.__create_button("Add (A)", add_callback, self.btn_img_add),
                            1, 0
        )
        self.__place_button(self.__create_button("Delete (X)", delete_callback, self.btn_img_delete),
                            1, 1
        )
        self.__place_button(self.__create_button("Settings (S)", settings_callback, self.btn_img_settings),
                            2, 0
        )
        self.__place_button(self.__create_button("Help", help_callback, self.btn_img_help),
                            2, 1
        )
        self.__place_button(self.__create_button("Recalculate (R)", recalculate_callback, self.btn_img_recalculate),
                            2, 2
        )

    def __create_button(self,
                        text: str,
                        callable: Callable[[], None],
                        image: tk.PhotoImage) -> ttk.Button:

        button = ttk.Button(self,
                            text=text,
                            compound=tk.TOP,
                            command=callable,
                            image=image)

        return button

    def __place_button(self,
                       button: tk.Widget,
                       column: int,
                       row: int) -> None:

        button.grid(
            column=column,
            row=row,
            padx=4,
            pady=4,
        )
