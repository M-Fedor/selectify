import matplotlib.pyplot as plt


def draw_histogram(data: dict) -> None:
    keys = sorted(data.keys())
    values = [data[k] for k in keys]

    plt.bar(keys, values)
    plt.show()