from typing import Callable, List, Dict, Optional, Set
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk
from os.path import join as joinpath
from field_element import ElementBase, PointSource, ChargePlane
import vectors
import settings
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


def _create_double_slider(master,
                          start: float,
                          end: float,
                          var: tk.DoubleVar,
                          resolution: float = 1,
                          start_label: Optional[str] = None,
                          end_label: Optional[str] = None) -> tk.Widget:

    if start_label is None:
        start_label = str(start)

    if end_label is None:
        end_label = str(end)

    frame = ttk.Frame(master)

    label_l = ttk.Label(
        frame,
        text=start_label
    )
    label_r = ttk.Label(
        frame,
        text=end_label
    )

    slider = tk.Scale(
        master=frame,
        from_=start,
        to=end,
        resolution=resolution,
        orient="horizontal",
        variable=var
    )

    label_l.grid(row=0, column=0)
    slider.grid(row=0, column=1)
    label_r.grid(row=0, column=2)

    return frame


def _create_int_slider(master,
                       start: int,
                       end: int,
                       var: tk.IntVar,
                       resolution: int = 1,
                       start_label: Optional[str] = None,
                       end_label: Optional[str] = None) -> tk.Widget:

    if start_label is None:
        start_label = str(start)

    if end_label is None:
        end_label = str(end)

    frame = ttk.Frame(master)

    label_l = ttk.Label(
        frame,
        text=start_label
    )
    label_r = ttk.Label(
        frame,
        text=end_label
    )

    slider = tk.Scale(
        master=frame,
        from_=start,
        to=end,
        resolution=resolution,
        orient="horizontal",
        variable=var
    )

    label_l.grid(row=0, column=0)
    slider.grid(row=0, column=1)
    label_r.grid(row=0, column=2)

    return frame


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
                 help_callback: Callable[[], None]):

        super().__init__()

        # Setup

        self.title(ControlsWindow.WINDOW_TITLE)

        self.__add_element_window: Optional[AddElementWindow] = None
        self.__settings_window: Optional[SettingsWindow] = None
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
        self.__place_button(self.__create_button("Settings (S)", self.__open_settings_window, _load_res_image(ControlsWindow.SETTINGS_BTN_IMG)),
                            2, 0
        )
        self.__place_button(self.__create_button("Help", help_callback, _load_res_image(ControlsWindow.HELP_BTN_IMG)),
                            2, 1
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

    def __open_settings_window(self) -> None:

        if self.__settings_window is None:
            self.__settings_window = SettingsWindow(
                self,
                destroy_callback=self.__clear_settings_window
            )
        else:
            self.__settings_window.lift()
            self.__settings_window.focus_set()

    def __clear_settings_window(self) -> None:
        self.__settings_window = None

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
        self.protocol("WM_DELETE_WINDOW", lambda: self.destroy_callback == self.destroy())  # If user manually tries to close the window

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

        strength_slider = _create_double_slider(
            master=strength_frame,
            start=AddElementWindow.STRENGTH_MIN,
            end=AddElementWindow.STRENGTH_MAX,
            var=self.__strength_var_raw
        )

        strength_label.pack(side=tk.TOP)
        strength_slider.pack(side=tk.TOP)

        strength_frame.pack(side=tk.TOP)

        # Angle slider

        angle_frame = ttk.Frame(master)

        angle_label = ttk.Label(angle_frame, text="Angle")

        angle_slider = _create_double_slider(
            master=angle_frame,
            start=0,
            end=2*np.pi,
            resolution=0.01,
            var=self.__angle_var_raw,
            end_label="2π"
        )

        angle_label.pack(side=tk.TOP)
        angle_slider.pack(side=tk.TOP)

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


class SettingsWindow(tk.Toplevel):

    def __init__(self,
                 master,
                 destroy_callback: Callable[[], None]):

        super().__init__(master, takefocus=True)

        self.double_vcmd = self.register(self.__double_callback)
        self.int_vcmd = self.register(self.__int_callback)

        self.destroy_callback = destroy_callback
        self.protocol("WM_DELETE_WINDOW", lambda: self.destroy_callback == self.destroy())  # If user manually tries to close the window

        self.__added_setting_index: int = 0
        self.__setting_labels: Set[tk.Widget] = set()
        self.__setting_widgets: Set[tk.Widget] = set()

        self.show_field_lines = tk.BooleanVar(self, settings.show_field_line_arrows)
        self.__create_bool_setting(
            "Show field line arrows",
            on_value_update=self.__update_show_field_lines,
            var=self.show_field_lines
        )

        self.line_count_factor = tk.DoubleVar(self, settings.field_line_count_factor)
        self.__create_bounded_double_setting(
            "Line count factor",
            on_value_update=self.__update_line_count_factor,
            var=self.line_count_factor,
            start=4.0,
            end=32.0,
            resolution=4.0
        )

        self.simulation_step_distance = tk.DoubleVar(self, settings.field_line_trace_step_distance_screen_space)
        self.__create_bounded_double_setting(
            "Simulation step distance",
            on_value_update=self.__update_simulation_step_distance,
            var=self.simulation_step_distance,
            start=0.5,
            end=20.0,
            resolution=0.5
        )

        self.maximum_line_steps = tk.IntVar(self, settings.field_line_trace_max_step_count)
        self.__create_input_int_setting(
            "Maximum simulation steps",
            on_value_update=self.__update_maximum_line_steps,
            var=self.maximum_line_steps
        )

        self.element_stop_distance = tk.DoubleVar(self, settings.field_line_trace_element_stop_distance_screen_space)
        self.__create_bounded_double_setting(
            "Line-element termination range",
            on_value_update=self.__update_element_stop_distance,
            var=self.element_stop_distance,
            start=0.0,
            end=10.0,
            resolution=0.5
        )

    def __int_callback(self, P):
        try:
            x = int(P)
            return True
        except ValueError:
            return False

    def __double_callback(self, P):
        try:
            x = float(P)
            return True
        except ValueError:
            return False

    def __update_show_field_lines(self):
        settings.show_field_line_arrows = self.show_field_lines.get()

    def __update_line_count_factor(self):
        settings.field_line_count_factor = self.line_count_factor.get()

    def __update_simulation_step_distance(self):
        settings.field_line_trace_step_distance_screen_space = self.simulation_step_distance.get()

    def __update_maximum_line_steps(self):
        settings.field_line_trace_max_step_count = self.maximum_line_steps.get()

    def __update_element_stop_distance(self):
        settings.field_line_trace_element_stop_distance_screen_space = self.element_stop_distance.get()

    def __create_bool_setting(self,
                              name: str,
                              on_value_update: Callable[[], None],
                              var: tk.BooleanVar) -> tk.BooleanVar:
        """Adds a boolean setting to the window and returns the variable created"""

        var.trace_add("write", lambda a,b,c: on_value_update())

        widget = tk.Checkbutton(self, variable=var)

        self.__setting_widgets.add(widget)

        self.__add_setting(name, widget)

        return var

    def __create_bounded_double_setting(self,
                                        name: str,
                                        on_value_update: Callable[[], None],
                                        var: tk.DoubleVar,
                                        start: float,
                                        end: float,
                                        resolution: float=1,
                                        start_label: Optional[str] = None,
                                        end_label: Optional[str] = None) -> tk.DoubleVar:
        """Adds a floating-point setting with a slider to the window and returns the variable created"""

        var.trace_add("write", lambda a,b,c: on_value_update())

        widget = _create_double_slider(
            master=self,
            start=start,
            end=end,
            var=var,
            resolution=resolution,
            start_label=start_label,
            end_label=end_label
        )

        self.__setting_widgets.add(widget)

        self.__add_setting(name, widget)

        return var

    def __create_input_double_setting(self,
                                      name: str,
                                      on_value_update: Callable[[], None],
                                      var: tk.DoubleVar) -> tk.DoubleVar:
        """Adds a floating-point setting using a textbox input to the window and returns the variable created"""

        var.trace_add("write", lambda a,b,c: on_value_update())

        widget = ttk.Entry(
            master=self,
            textvariable=var,
            validate="all",
            validatecommand=(self.double_vcmd, "%P")
        )

        self.__setting_widgets.add(widget)

        self.__add_setting(name, widget)

        return var

    def __create_bounded_int_setting(self,
                                     name: str,
                                     on_value_update: Callable[[], None],
                                     var: tk.IntVar,
                                     start: int,
                                     end: int,
                                     resolution: int=1,
                                     start_label: Optional[str] = None,
                                     end_label: Optional[str] = None) -> tk.IntVar:
        """Adds an integer setting with a slider to the window and returns the variable created"""

        var.trace_add("write", lambda a,b,c: on_value_update())

        widget = _create_int_slider(
            master=self,
            start=start,
            end=end,
            var=var,
            resolution=resolution,
            start_label=start_label,
            end_label=end_label
        )

        self.__setting_widgets.add(widget)

        self.__add_setting(name, widget)

        return var

    def __create_input_int_setting(self,
                                   name: str,
                                   on_value_update: Callable[[], None],
                                   var: tk.IntVar) -> tk.IntVar:
        """Adds an integer setting using a textbox input to the window and returns the variable created"""

        var.trace_add("write", lambda a,b,c: on_value_update())

        widget = ttk.Entry(
            master=self,
            textvariable=var,
            validate="all",
            validatecommand=(self.int_vcmd, "%P")
        )

        self.__setting_widgets.add(widget)

        self.__add_setting(name, widget)

        return var

    def __add_setting(self,
                      label: str,
                      setting_widget: tk.Widget) -> None:

        label_widget = tk.Label(self, text=label)
        self.__setting_labels.add(label_widget)

        label_widget.grid(
            row=self.__added_setting_index,
            column=0,
            padx=10,
            pady=5
        )
        setting_widget.grid(
            row=self.__added_setting_index,
            column=1,
            padx=10,
            pady=5
        )

        self.__added_setting_index += 1
