from .read_label_files import read_label_files
from .read_image_files import read_image_files

def dataset():
    images = read_image_files("data/train-images.idx3-ubyte")
    labels = read_label_files("data/train-labels.idx1-ubyte")

    return images, labels
