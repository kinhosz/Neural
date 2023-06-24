from tests.test_learn_rate import test_learn_rate_cpu

tests = [test_learn_rate_cpu]

for t in tests:
    t()
    print("{}: OK".format(t.__name__))