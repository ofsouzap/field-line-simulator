#!/bin/env python3

from visualisation_window import create_window as create_visualisation_window


def main():

    visualisation_controller = create_visualisation_window()

    visualisation_controller.run_app()


if __name__ == "__main__":
    main()
