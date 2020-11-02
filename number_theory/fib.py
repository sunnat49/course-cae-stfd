import sys


def fib(n):
    if n < 0:
        raise ValueError('Fibonacci number subscript is negative')

    a, b = 0, 1
    if n == 0:
        return a

    for i in range(n - 1):
        a, b = b, a + b

    return b


def fib_mod(n, m):
    return fib(n) % m


def main(argv):
    with open(argv[1], 'r') as input_file:
        for line in input_file:
            if not line.strip():
                continue

            n, m = map(int, line.split())
            print(fib_mod(n, m))


if __name__ == '__main__':
    main(sys.argv)
