from typing import Dict, Callable, Union, List, Tuple
from pyglet.window import key as Key
from pyglet.window.key import MOD_SHIFT, MOD_CTRL, MOD_ALT


RawCommand = Union[str, int, Tuple[str, int], Tuple[int, int]]
Command = Tuple[int, int]


__char_keys: Dict[str, int] = {
    "A": Key.A,
    "B": Key.B,
    "C": Key.C,
    "D": Key.D,
    "E": Key.E,
    "F": Key.F,
    "G": Key.G,
    "H": Key.H,
    "I": Key.I,
    "J": Key.J,
    "K": Key.K,
    "L": Key.L,
    "M": Key.M,
    "N": Key.N,
    "O": Key.O,
    "P": Key.P,
    "Q": Key.Q,
    "R": Key.R,
    "S": Key.S,
    "T": Key.T,
    "U": Key.U,
    "V": Key.V,
    "W": Key.W,
    "X": Key.X,
    "Y": Key.Y,
    "Z": Key.Z,
}


def _char_to_key(c: str) -> int:

    c = c.upper()

    if c in __char_keys:
        return __char_keys[c]
    else:
        raise ValueError(c)


class Shortcuts:

    def __init__(self):

        self.__shortcuts: Dict[Command, List[Callable[[], None]]] = {}

    @staticmethod
    def __process_cmd(raw: RawCommand) -> Command:

        match raw:

            case str() as c:
                return (_char_to_key(c), 0)

            case int() as key:
                return (key, 0)

            case (str() as c, int() as mods):
                return (_char_to_key(c), mods)

            case (int(), int()):
                return raw

    def add_shortcut(self,
                     _cmd: RawCommand,
                     callback: Callable[[], None]) -> None:

        cmd = self.__process_cmd(_cmd)

        if cmd in self.__shortcuts:
            self.__shortcuts[cmd].append(callback)
        else:
            self.__shortcuts[cmd] = [callback]

    def use_shortcuts(self,
                      _cmd: RawCommand) -> None:

        cmd = self.__process_cmd(_cmd)

        if cmd in self.__shortcuts:
            for callback in self.__shortcuts[cmd]:
                callback()
