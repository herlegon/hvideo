from enum import IntEnum


class ColorCode(IntEnum):
    red = 31
    green = 32
    orange = 33
    blue = 34
    purple = 35
    cyan = 36
    lightgrey = 37
    darkgrey = 90
    lighred = 91
    lightgreen = 92
    yellow = 93
    lightblue = 94
    pink = 95
    lightcyan = 96
    white = 97


def function_generator(color:ColorCode) -> callable:
    def _function(*values: object) -> str:
        return _color_str_template(color).format(values[0])
    return _function


def _color_str_template(color:ColorCode) -> str:
    return "\033[%dm{}\033[00m" % (color.value)


def red(*values: object) -> str:
    return _color_str_template(ColorCode.red).format(values[0])

def green(*values: object) -> str:
    return _color_str_template(ColorCode.green).format(values[0])

def orange(*values: object) -> str:
    return _color_str_template(ColorCode.orange).format(values[0])

def blue(*values: object) -> str:
    return _color_str_template(ColorCode.blue).format(values[0])

def purple(*values: object) -> str:
    return _color_str_template(ColorCode.purple).format(values[0])

def cyan(*values: object) -> str:
    return _color_str_template(ColorCode.cyan).format(values[0])

def lightgrey(*values: object) -> str:
    return _color_str_template(ColorCode.lightgrey).format(values[0])

def darkgrey(*values: object) -> str:
    return _color_str_template(ColorCode.darkgrey).format(values[0])

def lighred(*values: object) -> str:
    return _color_str_template(ColorCode.lighred).format(values[0])

def lightgreen(*values: object) -> str:
    return _color_str_template(ColorCode.lightgreen).format(values[0])

def yellow(*values: object) -> str:
    return _color_str_template(ColorCode.yellow).format(values[0])

def lightblue(*values: object) -> str:
    return _color_str_template(ColorCode.lightblue).format(values[0])

def pink(*values: object) -> str:
    return _color_str_template(ColorCode.pink).format(values[0])

def lightcyan(*values: object) -> str:
    return _color_str_template(ColorCode.lightcyan).format(values[0])

def white(*values: object) -> str:
    return _color_str_template(ColorCode.white).format(values[0])


__all__ = [
    "red",
    "green",
    "orange",
    "blue",
    "purple",
    "cyan",
    "lightgrey",
    "darkgrey",
    "lighred",
    "lightgreen",
    "yellow",
    "lightblue",
    "pink",
    "lightcyan",
    "white",
]
