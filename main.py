#!/bin/env python3

from typing import Optional
from threading import Thread
import numpy as np
from field import Field
from field_element import ElementBase
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

        new_ele = self.add_config.create_element(np.array([x, y], dtype=float))

        controller.add_field_element(new_ele)

        if settings.auto_recalcualate:
            controller.recalculate()

    def __delete_click(self, x: int, y: int, controller: VisualisationController) -> None:

        ele_removed = controller.try_delete_field_element_at(x, y)

        if ele_removed and settings.auto_recalcualate:
            controller.recalculate()


def main():

    # General setup

    click_mode = ClickMode()

    # Create windows

    visualisation_controller = create_visualisation_window(on_mouse_press=lambda x, y, btn, mods: click_mode.on_click(x, y, visualisation_controller))

    controls_window = ControlsWindow(
        save_callback=lambda: print("Save pressed"),
        load_callback=lambda: print("Load pressed"),
        set_add_config_callback=lambda config: click_mode.set_mode_add(config),
        delete_callback=click_mode.set_mode_delete,
        settings_callback=lambda: print("Settings pressed"),
        help_callback=lambda: print("Help pressed"),
        recalculate_callback=lambda: print("Recalculate pressed")
    )

    # Create visualisation window thread

    visualisation_thread = Thread(target=visualisation_controller.run_app)

    # Run windows
    # N.B. tkinter doesn't like not being on the main thread so I run it on the main thread

    visualisation_thread.start()
    controls_window.mainloop()

    visualisation_thread.join()


if __name__ == "__main__":
    main()
