#!/bin/env python3

from typing import Optional, Callable
from threading import Thread
import numpy as np
from webbrowser import open as webbrowser_open
from visualisation_window import create_window as create_visualisation_window
from visualisation_window import Controller as VisualisationController
from menu_windows import ControlsWindow, AddElementWindow
from field_file_windows import save_field, load_field
from shortcuts import Shortcuts, MOD_CTRL, MOD_SHIFT, MOD_ALT
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
        self.__shortcuts = Shortcuts()

        # Create windows

        self.visualisation_controller = create_visualisation_window(
            on_exit=self.quit,
            on_mouse_press=self._visualisation_clicked,
            on_key_press=self.__shortcuts.use_shortcuts
        )

        self.controls_window = ControlsWindow(
            on_exit=self.quit,
            on_char_press=lambda cmd: self.__shortcuts.use_shortcuts(cmd),
            save_callback=self.save,
            load_callback=self.load,
            set_add_config_callback=self.set_click_mode_add,
            delete_callback=self.set_click_mode_delete,
            help_callback=self.show_help
        )

        # Set up shortcuts

        self.__shortcuts.add_shortcut(("S", MOD_CTRL), self.save)
        self.__shortcuts.add_shortcut(("O", MOD_CTRL), self.load)
        self.__shortcuts.add_shortcut("A", self.controls_window.open_add_elements_window)
        self.__shortcuts.add_shortcut("X", self.set_click_mode_delete)
        self.__shortcuts.add_shortcut("S", self.controls_window.open_settings_window)

    def run(self):

        # Create visualisation window thread

        self._visualisation_thread = Thread(target=lambda: self.visualisation_controller.run_app())

        # Run windows
        # N.B. tkinter doesn't like not being on the main thread so I run it on the main thread

        self._visualisation_thread.start()

        self.controls_window.after(100, self.visualisation_controller.activate_window)
        self.controls_window.mainloop()

        self._visualisation_thread.join()

    def _visualisation_clicked(self, x, y, btn, mods):
        self.__click_mode.on_click(x, y, self.visualisation_controller)

    def quit(self):
        self.controls_window.try_exit()
        self.visualisation_controller.quit_app()

    def save(self):

        field = self.visualisation_controller.get_field()
        save_field(field)

    def load(self):

        field = load_field()

        if field is not None:

            self.visualisation_controller.set_field(field)
            self.recalculate()

    def set_click_mode_add(self, config: AddConfig):
        self.visualisation_controller.set_click_mode_add()
        self.__click_mode.set_mode_add(config)

    def set_click_mode_delete(self):
        self.visualisation_controller.set_click_mode_delete()
        self.__click_mode.set_mode_delete()

    def open_settings(self):
        print("Settings pressed")  # TODO - proper functionality

    def show_help(self):
        webbrowser_open("https://github.com/ofsouzap/field-line-simulator/wiki/Instructions")

    def recalculate(self):
        self.visualisation_controller.recalculate()


def main():
    main_controller = MainController()
    main_controller.run()


if __name__ == "__main__":
    main()
