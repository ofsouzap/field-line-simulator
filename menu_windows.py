from typing import Callable, List, Dict
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk
from os.path import join as joinpath
from field_element import ElementBase, PointSource
import numpy as np


def _resource(*path: str) -> str:
    return joinpath("resources", *path)


__loaded_images: Dict[str, tk.PhotoImage] = {}
def _load_res_image(res_path: str) -> tk.PhotoImage:
    """Loads an image from a resources path. N.B. "resources/" shouldn't be included in the res_path provided"""

    global __loaded_images

    if res_path in __loaded_images:
        return __loaded_images[res_path]
    else:
        img = tk.PhotoImage(file=_resource(res_path))
        __loaded_images[res_path] = img
        return img


class ControlsWindow(tk.Tk):

    WINDOW_TITLE = "Fields - Controls"

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

        # Create buttons

        self.__place_button(self.__create_button("Save (Ctrl+S)", save_callback, _load_res_image(ControlsWindow.SAVE_BTN_IMG)),
                            0, 0
        )
        self.__place_button(self.__create_button("Open (Ctrl+O)", load_callback, _load_res_image(ControlsWindow.LOAD_BTN_IMG)),
                            0, 1
        )
        self.__place_button(self.__create_button("Add (A)", add_callback, _load_res_image(ControlsWindow.ADD_BTN_IMG)),
                            1, 0
        )
        self.__place_button(self.__create_button("Delete (X)", delete_callback, _load_res_image(ControlsWindow.DELETE_BTN_IMG)),
                            1, 1
        )
        self.__place_button(self.__create_button("Settings (S)", settings_callback, _load_res_image(ControlsWindow.SETTINGS_BTN_IMG)),
                            2, 0
        )
        self.__place_button(self.__create_button("Help", help_callback, _load_res_image(ControlsWindow.HELP_BTN_IMG)),
                            2, 1
        )
        self.__place_button(self.__create_button("Recalculate (R)", recalculate_callback, _load_res_image(ControlsWindow.RECALCULATE_BTN_IMG)),
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


class AddElementWindow(tk.Tk):

    WINDOW_TITLE = "Fields - Add Element"

    ELEMENT_ICON_SIZE = 32

    STRENGTH_MIN = -50
    STRENGTH_STEP = 1
    STRENGTH_MAX = 50

    class __Element(ABC):

        def load_icon(self) -> tk.PhotoImage:
            return _load_res_image(self.get_icon_path())

        @abstractmethod
        def get_display_name(self) -> str:
            """Gets the name that should be displayed for the element in the menu"""
            raise NotImplementedError()

        @abstractmethod
        def get_icon_path(self) -> str:
            """Gets the resources path for the element's icon"""
            raise NotImplementedError()

        @abstractmethod
        def create_instance(self, pos: np.ndarray, strength: float) -> ElementBase:
            """Creates an instance of the element"""
            raise NotImplementedError()

    class __PointSource(__Element):
        def get_display_name(self) -> str: return "Point Source"
        def get_icon_path(self) -> str: return joinpath("element_icons", "placeholder.png")
        def create_instance(self, pos: np.ndarray, strength: float) -> ElementBase: return PointSource(pos, strength)

    class Config:
        def __init__(
            self,
            strength: float = 0
        ):
            self.strength = strength

    def __init__(self):

        super().__init__()

        # Setup

        self.title(AddElementWindow.WINDOW_TITLE)

        self.__elements_items: List[tk.Widget] = []

        self.__strength_var_raw = tk.DoubleVar(self)

        # Create frames

        self.elements_frame = tk.Frame(self)
        self.config_frame = tk.Frame(self)

        # Populate frames

        self.__populate_elements_frame([
            AddElementWindow.__PointSource()
        ])
        self.__populate_config_frame()

        # Layout

        self.elements_frame.pack(side=tk.LEFT, padx=15, pady=15)
        self.config_frame.pack(side=tk.RIGHT, padx=15, pady=15)

    @property
    def strength_config_val(self) -> float:
        return round(self.__strength_var_raw.get(), 0)

    def __populate_elements_frame(self,
                                  elements: List[__Element]) -> None:

        for ele in elements:
            _e = ele
            item = self.__create_element_item(
                self.elements_frame,
                ele,
                lambda: print(f"{_e.get_display_name()} selected")  # TODO
            )
            self.__elements_items.append(item)
            item.pack(side=tk.TOP)

    def __create_element_item(self,
                              master,
                              element: __Element,
                              callback: Callable[[], None]) -> tk.Widget:

        ele_img = element.load_icon()

        button = ttk.Button(
            master,
            text=element.get_display_name(),
            image=ele_img,
            compound=tk.RIGHT,
            command=callback
        )

        return button

    def __populate_config_frame(self) -> None:

        master = self.config_frame  # Just for convenience

        # Strength slider

        strength_frame = ttk.Frame(master)

        strength_label = ttk.Label(strength_frame, text="Element Strength")

        strength_slider_frame = ttk.Frame(strength_frame)

        strength_slider_label_l = ttk.Label(strength_slider_frame, text=str(AddElementWindow.STRENGTH_MIN))
        strength_slider_slider = tk.Scale(
            strength_slider_frame,
            from_=AddElementWindow.STRENGTH_MIN,
            to=AddElementWindow.STRENGTH_MAX,
            orient="horizontal",
            variable=self.__strength_var_raw,
        )
        strength_slider_label_r = ttk.Label(strength_slider_frame, text=str(AddElementWindow.STRENGTH_MAX))

        strength_slider_label_l.grid(row=0, column=0)
        strength_slider_slider.grid(row=0, column=1)
        strength_slider_label_r.grid(row=0, column=2)

        strength_label.pack(side=tk.TOP)
        strength_slider_frame.pack(side=tk.TOP)

        strength_frame.pack(side=tk.TOP)

    def get_current_config(self) -> Config:

        strength = self.strength_config_val

        config = AddElementWindow.Config(
            strength=strength
        )

        return config
