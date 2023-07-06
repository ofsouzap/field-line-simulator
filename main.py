#!/bin/env python3

import pyglet
from gui import open_gui


pyglet.resource.path = ["resources"]
pyglet.resource.reindex()


if __name__ == "__main__":
    open_gui()
