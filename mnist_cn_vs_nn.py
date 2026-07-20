import random
import time

from matplotlib import pyplot as plt

from Kinho import Neural, ConvNeural, Tensor
from tests.common.read_image_files import read_image_files
from tests.common.read_label_files import read_label_files
from tests.common.greaterIdx import greaterIdx
from tests.common.densityArr import densityArr

TRAIN_SIZE = 200
TEST_SIZE = 50
EVAL_EVERY = 20
N_CLASSES = 10

random.seed(42)


def load_mnist():
    images = read_image_files("data/train-images.idx3-ubyte")
    labels = read_label_files("data/train-labels.idx1-ubyte")

    zipped = list(zip(images, labels))
    random.shuffle(zipped)

    train = zipped[:TRAIN_SIZE]
    test = zipped[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    return train, test


def normalize(p):
    return (p / 255.0) * 2 - 1  # [0,255] -> [-1,1], background (0) -> -1, not 0


def to_flat(image):
    return [normalize(p) for p in image]


def to_tensor(image):
    t = Tensor((1, 28, 28))
    for i in range(28):
        for j in range(28):
            t[0][i][j] = normalize(image[i * 28 + j])
    return t


def eval_nn(robot, test):
    hits = 0
    for image, label in test:
        out = robot.send(to_flat(image))
        if greaterIdx(out) == label:
            hits += 1
    return hits / len(test)


def eval_cn(robot, test):
    hits = 0
    for image, label in test:
        out = robot.send(to_tensor(image))
        out_list = [out[i] for i in range(out.shape()[0])]
        if greaterIdx(out_list) == label:
            hits += 1
    return hits / len(test)


def train_nn(train, test):
    robot = Neural(sizes=[28 * 28, 128, N_CLASSES], eta=0.01, mini_batch_size=1)

    samples_seen = [0]
    accs = [eval_nn(robot, test)]

    t0 = time.perf_counter()
    for i, (image, label) in enumerate(train):
        robot.learn(to_flat(image), densityArr(label, N_CLASSES))

        if (i + 1) % EVAL_EVERY == 0:
            acc = eval_nn(robot, test)
            samples_seen.append(i + 1)
            accs.append(acc)
            print("[NN] {}/{} samples -> acc={:.3f}".format(i + 1, len(train), acc))
    elapsed = time.perf_counter() - t0
    print("[NN] total training time: {:.1f}s".format(elapsed))

    return samples_seen, accs


def train_cn(train, test):
    robot = ConvNeural(in_shape=(1, 28, 28), filters=[4, 8, 16, 32, 32], fc_layers=[64, N_CLASSES])

    samples_seen = [0]
    accs = [eval_cn(robot, test)]

    t0 = time.perf_counter()
    for i, (image, label) in enumerate(train):
        target = Tensor().load_from(densityArr(label, N_CLASSES))
        robot.learn(to_tensor(image), target)

        if (i + 1) % EVAL_EVERY == 0:
            acc = eval_cn(robot, test)
            samples_seen.append(i + 1)
            accs.append(acc)
            print("[CN] {}/{} samples -> acc={:.3f}".format(i + 1, len(train), acc))
    elapsed = time.perf_counter() - t0
    print("[CN] total training time: {:.1f}s".format(elapsed))

    return samples_seen, accs


def main():
    train, test = load_mnist()
    print("train={} test={}".format(len(train), len(test)))

    nn_x, nn_y = train_nn(train, test)
    cn_x, cn_y = train_cn(train, test)

    plt.plot(nn_x, nn_y, label="Neural (NN)", marker='o')
    plt.plot(cn_x, cn_y, label="ConvNeural (CN)", marker='o')
    plt.xlabel("samples seen")
    plt.ylabel("accuracy")
    plt.title("MNIST: NN x CN ({} train / {} test)".format(TRAIN_SIZE, TEST_SIZE))
    plt.legend()
    plt.savefig("mnist_cn_vs_nn.png")
    print("plot saved to mnist_cn_vs_nn.png")


if __name__ == "__main__":
    main()
