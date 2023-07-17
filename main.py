#!/bin/env python3

from threading import Thread
from visualisation_window import create_window as create_visualisation_window
from menu_windows import ControlsWindow


def main():

    # Create windows

    visualisation_controller = create_visualisation_window()

    controls_window = ControlsWindow(
        save_callback=lambda: print("Save pressed"),
        load_callback=lambda: print("Load pressed"),
        delete_callback=lambda: print("Delete pressed"),
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
