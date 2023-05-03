from tests.correctness import test as correctness
from tests.speed import test as speed

TESTS = [correctness, speed]

for T in TESTS:
    T()