#!/bin/env python3

from gui import open_gui
import pyglet


# Set-up pyglet resources

pyglet.resource.path = ["resources"]
pyglet.resource.reindex()


# Run main gui

if __name__ == "__main__":
    open_gui()
