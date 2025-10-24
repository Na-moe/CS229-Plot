from .dataset_example import plot_dataset_example
from .dataset_discrete import plot_dataset_discrete


def plot_all():
    plot_dataset_example()
    plot_dataset_discrete()


if __name__ == "__main__":
    plot_all()
