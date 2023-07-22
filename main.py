#!/bin/env python3

from typing import Optional, Callable
from threading import Thread
import numpy as np
from visualisation_window import create_window as create_visualisation_window
from visualisation_window import Controller as VisualisationController
from menu_windows import ControlsWindow, AddElementWindow
import settings


AddConfig = AddElementWindow.Config


class ClickMode:

    MODE_NONE = 0
    MODE_ADD = 1
    MODE_DELETE = 2

    def __init__(self, mode=MODE_NONE):

        self.mode = mode
        self.add_config: Optional[AddConfig] = None

    def set_mode_none(self) -> None:
        self.mode = ClickMode.MODE_NONE

    def set_mode_add(self, config: AddConfig) -> None:
        self.mode = ClickMode.MODE_ADD
        self.add_config = config

    def set_mode_delete(self) -> None:
        self.mode = ClickMode.MODE_DELETE

    def on_click(self,
                 x: int,
                 y: int,
                 controller: VisualisationController) -> None:

        if self.mode == ClickMode.MODE_NONE:
            return
        elif self.mode == ClickMode.MODE_ADD:
            self.__add_click(x, y, controller)
        elif self.mode == ClickMode.MODE_DELETE:
            self.__delete_click(x, y, controller)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __add_click(self, x: int, y: int, controller: VisualisationController) -> None:

        assert self.add_config is not None, "Trying to add element without setting add_config"

        new_ele = self.add_config.create_element(np.array([x*settings.VIEWPORT_SCALE_FAC, y*settings.VIEWPORT_SCALE_FAC], dtype=float))

        controller.add_field_element(new_ele)

        if settings.auto_recalcualate:
            controller.recalculate()

    def __delete_click(self, x: int, y: int, controller: VisualisationController) -> None:

        ele_removed = controller.try_delete_field_element_at(x, y)

        if ele_removed and settings.auto_recalcualate:
            controller.recalculate()


class MainController:

    def __init__(self):

        # General setup

        self.__click_mode = ClickMode()

        # Create windows

        self.visualisation_controller = create_visualisation_window(on_mouse_press=self._visualisation_clicked)

        self.controls_window = ControlsWindow(
            save_callback=self.save,
            load_callback=self.load,
            set_add_config_callback=self.set_click_mode_add,
            delete_callback=self.set_click_mode_delete,
            settings_callback=self.open_settings,
            help_callback=self.show_help,
            recalculate_callback=self.recalculate
        )

    def run(self):

        # Create visualisation window thread

        self._visualisation_thread = Thread(target=self.visualisation_controller.run_app)

        # Run windows
        # N.B. tkinter doesn't like not being on the main thread so I run it on the main thread

        self._visualisation_thread.start()
        self.controls_window.mainloop()

        self._visualisation_thread.join()

    def _visualisation_clicked(self, x, y, btn, mods):
        self.__click_mode.on_click(x, y, self.visualisation_controller)

    def save(self):
        print("Save pressed")  # TODO - proper functionality

    def load(self):
        print("Load pressed")  # TODO - proper functionality

    def set_click_mode_add(self, config: AddConfig):
        self.visualisation_controller.set_click_mode_add()
        self.__click_mode.set_mode_add(config)

    def set_click_mode_delete(self):
        self.visualisation_controller.set_click_mode_delete()
        self.__click_mode.set_mode_delete()

    def open_settings(self):
        print("Settings pressed")  # TODO - proper functionality

    def show_help(self):
        print("Help pressed")  # TODO - proper functionality

    def recalculate(self):
        print("Recalculate pressed")  # TODO - proper functionality


def main():
    main_controller = MainController()
    main_controller.run()


if __name__ == "__main__":
    main()
