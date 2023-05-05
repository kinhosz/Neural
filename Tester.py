from tests.correctness import test as correctness
from tests.speed import test as speed
from tests.network import test as network

TESTS = [network, correctness, speed]

for T in TESTS:
    T()