import matplotlib.pyplot as plt

from .utils import get_length, write_binary


def draw_histogram(data: dict) -> None:
    keys = sorted(data.keys())
    values = [data[k] for k in keys]

    plt.bar(keys, values)
    plt.show()


def write_file(data: dict, path: str) -> None:
    file = open(path, 'wb')

    for key, value in data.items():
        write_binary(file, key, get_length(key))
        write_binary(file, value, get_length(value))
