from typing import Callable, List, Dict, Optional
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk
from copy import copy
from os.path import join as joinpath
from field_element import ElementBase, PointSource, ChargePlane
import vectors
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

    SAVE_BTN_IMG = joinpath("controls_icons", "save.png")
    LOAD_BTN_IMG = joinpath("controls_icons", "load.png")
    ADD_BTN_IMG = joinpath("controls_icons", "add.png")
    DELETE_BTN_IMG = joinpath("controls_icons", "delete.png")
    SETTINGS_BTN_IMG = joinpath("controls_icons", "placeholder.png")
    HELP_BTN_IMG = joinpath("controls_icons", "placeholder.png")
    RECALCULATE_BTN_IMG = joinpath("controls_icons", "placeholder.png")

    def __init__(self,
                 save_callback: Callable[[], None],
                 load_callback: Callable[[], None],
                 set_add_config_callback: Callable[["AddElementWindow.Config"], None],
                 delete_callback: Callable[[], None],
                 settings_callback: Callable[[], None],
                 help_callback: Callable[[], None],
                 recalculate_callback: Callable[[], None]):

        super().__init__()

        # Setup

        self.title(ControlsWindow.WINDOW_TITLE)

        self.__add_element_window: Optional[AddElementWindow] = None
        self.__set_add_config_callback = set_add_config_callback

        self.button_frame = tk.Frame(self)

        # Create buttons

        self.__place_button(self.__create_button("Save (Ctrl+S)", save_callback, _load_res_image(ControlsWindow.SAVE_BTN_IMG)),
                            0, 0
        )
        self.__place_button(self.__create_button("Open (Ctrl+O)", load_callback, _load_res_image(ControlsWindow.LOAD_BTN_IMG)),
                            0, 1
        )
        self.__place_button(self.__create_button("Add (A)", self.__open_add_elements_window, _load_res_image(ControlsWindow.ADD_BTN_IMG)),
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

    def __open_add_elements_window(self) -> None:

        if self.__add_element_window is None:
            self.__add_element_window = AddElementWindow(
                self,
                select_callback=self.__set_add_config_callback,
                destroy_callback=self.__clear_add_element_window
            )
        else:
            self.__add_element_window.lift()
            self.__add_element_window.focus_set()

    def __clear_add_element_window(self) -> None:
        self.__add_element_window = None

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


class AddElementWindow(tk.Toplevel):

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
        def create_instance(self, pos: np.ndarray, config: "AddElementWindow.Config") -> ElementBase:
            """Creates an instance of the element"""
            raise NotImplementedError()

    class __PointSource(__Element):
        def get_display_name(self) -> str: return "Point Source"
        def get_icon_path(self) -> str: return joinpath("element_icons", "point_source.png")
        def create_instance(self, pos: np.ndarray, config: "AddElementWindow.Config") -> ElementBase: return PointSource(pos, config.strength)

    class __ChargePlane(__Element):
        def get_display_name(self) -> str: return "Charge Plane"
        def get_icon_path(self) -> str: return joinpath("element_icons", "plane.png")
        def create_instance(self, pos: np.ndarray, config: "AddElementWindow.Config") -> ElementBase:
            return ChargePlane(
                pos,
                vectors.angle_to_vec2d(config.angle + (np.pi/2)),
                    # Offset by π/2 so that the "angle" represents the angle of the plane, not the normal of the plane
                config.strength
            )

    class Config:
        def __init__(
            self,
            strength: float,
            angle: float,
            element_gen: Callable[[np.ndarray, "AddElementWindow.Config"], ElementBase]
        ):
            self.strength = strength
            self.angle = angle
            self.create_element: Callable[[np.ndarray], ElementBase] = lambda pos: element_gen(pos, self)

    def __init__(self,
                 master,
                 select_callback: Callable[[Config], None],
                 destroy_callback: Callable[[], None]):

        super().__init__(master, takefocus=True)

        # Setup

        self.title(AddElementWindow.WINDOW_TITLE)

        self.select_callback = select_callback
        self.destroy_callback = destroy_callback

        self.__elements_items: List[tk.Widget] = []

        self.__strength_var_raw = tk.DoubleVar(self)
        self.__strength_var_raw.set(1)  # Default to 1

        self.__angle_var_raw = tk.DoubleVar(self)
        self.__angle_var_raw.set(0)  # Default to 0

        # Create frames

        self.elements_frame = tk.Frame(self)
        self.config_frame = tk.Frame(self)

        # Populate frames

        self.__populate_elements_frame([
            AddElementWindow.__PointSource(),
            AddElementWindow.__ChargePlane(),
        ])
        self.__populate_config_frame()

        # Layout

        self.elements_frame.pack(side=tk.LEFT, padx=15, pady=15)
        self.config_frame.pack(side=tk.RIGHT, padx=15, pady=15)

    @property
    def strength_config_val(self) -> float:
        return round(self.__strength_var_raw.get(), 0)

    @property
    def angle_config_val(self) -> float:
        return round(self.__angle_var_raw.get(), 2)

    def __populate_elements_frame(self,
                                  elements: List[__Element]) -> None:

        for ele in elements:

            # _e = copy(ele)  # Storing here allows the lambdas to keep their values through the iterations

            item = self.__create_element_item(
                self.elements_frame,
                ele,
                lambda e=ele: self.__select_element(e)
            )
            self.__elements_items.append(item)
            item.pack(side=tk.TOP)

    def __select_element(self, ele: __Element) -> None:

        config = self.__create_config(ele.create_instance)

        self.select_callback(config)

        self.destroy_callback()
        self.destroy()

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

        strength_label = ttk.Label(strength_frame, text="Strength")

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

        # Angle slider

        angle_frame = ttk.Frame(master)

        angle_label = ttk.Label(angle_frame, text="Angle")

        angle_slider_frame = ttk.Frame(angle_frame)

        angle_slider_label_l = ttk.Label(angle_slider_frame, text="0")
        angle_slider_slider = tk.Scale(
            angle_slider_frame,
            from_=0,
            to=2*np.pi,
            orient="horizontal",
            resolution=0.01,
            variable=self.__angle_var_raw,
        )
        angle_slider_label_r = ttk.Label(angle_slider_frame, text="2π")

        angle_slider_label_l.grid(row=0, column=0)
        angle_slider_slider.grid(row=0, column=1)
        angle_slider_label_r.grid(row=0, column=2)

        angle_label.pack(side=tk.TOP)
        angle_slider_frame.pack(side=tk.TOP)

        angle_frame.pack(side=tk.TOP)

    def __create_config(self, gen_ele: Callable[[np.ndarray, Config], ElementBase]) -> Config:

        strength = self.strength_config_val
        angle = self.angle_config_val

        config = AddElementWindow.Config(
            strength,
            angle,
            gen_ele
        )

        return config
