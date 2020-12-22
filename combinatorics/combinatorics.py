from itertools import permutations, product


pool = {
    'C': 2,
    'N': 1,
    'H': 6,
}

valency = {
    'C': 4,
    'O': 2,
    'N': 5
}

def generate_cores_with_len(pool, n):
    structural = [key
                  for key, val in pool.items() if key != 'H'
                  for i in range(val)]
    for core in permutations(structural, n):
        for mult in product(range(1,4), repeat=n-1):
            yield [j for em in zip(core, mult) for j in em] + [core[-1],]


def max_len(pool):
    # list comprehensions
    # dict items
    return sum(val for key, val in pool.items() if key != 'H')


def generate_cores(pool):
    for n in range(1, max_len(pool) + 1):
        yield from generate_cores_with_len(pool, n)


def finalize_compound(core):
    if len(core) == 1:
        elem = core[0]
        return [(elem, valency[elem]), ]

    leftmost_saturation = core[1]
    leftmost_hydration = valency[core[0]] - leftmost_saturation
    if (leftmost_hydration < 0):
        return None
    core[0] = (core[0], leftmost_hydration)

    rightmost_saturation = core[-2]
    rightmost_hydration = valency[core[-1]] - rightmost_saturation
    if rightmost_hydration < 0:
        return None
    core[-1] = (core[-1], rightmost_hydration)

    # for atom, ls, rs in zip(core[2:-1:2], core[1:-1:2], core[3:-1:2])
    for i in range(2, len(core) - 2):
        elem = core[i]
        ls = core[i-1]
        rs = core[i+1]
        hyd = valency[elem] - (ls + rs)
        if hyd < 0:
            return None
        core[i] = (elem, hyd)
    return core

def main():
    f_min =  1e+9
    f_max = -1e+9
    min_compound = None
    max_compound = None

    for core in generate_cores(pool):
        compound = finalize_compound(core)
        if compound is None:
            continue

        f_cur = f(compound)
        if (f_cur < f_min):
            f_min = f_cur
            min_compound = compound

        if (f_cur > f_max):
            f_max = f_cur
            max_compound = compound

if __name__ == '__main__':
    main()