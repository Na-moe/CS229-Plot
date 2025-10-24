from .activations import plot_activations
from .house_dataset_relu import plot_house_dataset_relu


def plot_all():
    plot_activations()
    plot_house_dataset_relu()


if __name__ == "__main__":
    plot_all()
