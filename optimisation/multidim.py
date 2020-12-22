"""
N точек размещены на единичной сфере (x^2+y^2+z^2=1).
d(A,B) - евклидово расстояние между точками A и B.

Найти размещения P={A1, A2,..., AN} для N (N >= 2) точек,
минимизирующие следующие функции (достаточно одного для каждой из функций):

1. F1(P) = - sum(d(Ai, Aj)^2 for i in 1..N for j in 1..N)
   Сумма квадратов расстояний между точками, взятая за знаком минус
2. F2(P) =   sum(1/d(Ai, Aj) for i in 1..N for j in 1..N if i != j)
   Сумма обратных расстояний между различными точками
3. F3(P) = - min(d(Ai, Aj)   for i in 1..N for j in 1..N if i != j)
   Минимум среди всех расстояний между различными точками, взятый со знаком
   минус

Координатные оси всегда можно повернуть так, что
1. Точка A1 будет иметь координаты (0, 0, 1);
2. Точка A2 будет находиться в плоскости Oyz, т.е. A2.x = 0.
Возьмём эти утверждения в качестве дополнительных условий.

Решить задачу для N=2. Известно аналитическое решение (полюса)
A1=(0,0,1), A2=(0,0,-1), одинаковое для всех фукнций.

Решить задачу для N=3. Известно аналитическое решение (равносторонний треугольник)
A1=(0,0,1), A2=(0,sqrt(3)/2,-1/2), A3=(0,-sqrt(3)/2,-1/2),
одинаковое для всех фукнций.

Решить задачу для N=4. Вероятное решение - вершины правильного тетраэдра.

Проверить, образуют ли решения для N=6, N=8, N=12, N=20 для какой-либо из
функций вершины правильных многогранников: октаэдра, куба, икосаэдра и
додекаэдра соответственно.

Подсказка:
    Kаждая точка на сфере описывается двумя углами phi и theta.
    Удобно взять:
        x = sin(theta)
        y = sin(phi)*cos(theta)
        z = cos(phi)*cos(theta)
    Целевые функции - это функции 2N аргументов, из которых 3 фиксированы:
    * Для точки A1 theta==0, phi==0
    * Для точки A2 theta==0

    В качестве начального приближения можно брать случайно размещенные точки.
"""
from math import sin, cos, asin, atan2
import math
import random
import numpy as np

# Поскольку функцию dist ввели только в Python 3.8, то для пользователей более
# старых версий придётся определить функцию dist самостоятельно.

if hasattr(math, 'dist'):
    dist = math.dist
else:
    def dist(p, q):
        math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))


def _to_args_gen(points):
    """ Генерирует phi1, theta2, phi2,..., theta(N-1), phi(N-1) """
    yield math.atan2(points[1][2], points[1][1])
    for point in points[2:]:
        yield math.asin(point[0])
        yield math.atan2(point[2], point[1])


def random_args_generator(N):
    if N < 2:
        raise ValueError

    yield random.uniform(-math.pi, math.pi)
    for i in range(2, N):
        yield random.uniform(-math.pi/2, math.pi/2)
        yield random.uniform(-math.pi, math.pi)


def _to_points_gen(args):
    """ Генерирует (x0,y0,z0), (x1,y1,z1), ..., (x(N-1), y(N-1), z(N-1)) """
    # первая точка
    yield (0, 0, 1)

    phi = args[0]
    # вторая точка
    yield (0, math.sin(phi), math.cos(phi))

    # остальные точки
    for theta, phi in zip(args[1::2], args[2::2]):
        yield (sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi))


def to_args(points):
    """ Возвращает [phi1, theta2, phi2,..., theta(N-1), phi(N-1)] """
    return list(_to_args_gen(points))


def to_points(args):
    """ Возвращает [(x0,y0,z0), (x1,y1,z1), ..., (x(N-1), y(N-1), z(N-1))] """
    return list(_to_points_gen(args))


def F1(args):
    points = to_points(args)
    return -sum(dist(point1, point2)**2 for point1 in points for point2 in points)


def F2(args):
    points = to_points(args)
    return sum(1/dist(point1, point2)
               for point1 in points for point2 in points
               if point1 is not point2)


def F3(args):
    points = to_points(args)
    return -min(dist(point1, point2)
                for point1 in points for point2 in points
                if point1 is not point2)


class MultidimensionalOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def optimize(self, objective, initial, gradient=None):
        """ Здесь пишете ваш любимый метод
        objective - целевая функция (минимизируемая)
        gradient - необязательная функция градиента
        """
        return initial


class NelderMead(MultidimensionalOptimizer):
    def __init__(self, alpha=1, gamma=2, rho=0.5, sigma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def precision_reached(self, simplex):
        return simplex.std(axis=0).mean() < 1e-2

    def sort_simplex(self, simplex, key):
        raise NotImplementedError

    def optimize(self, objective, initial_points):
        initial_points
        n = len(initial_points) - 1
        simplex = np.array(initial_points)

        while not self.precision_reached(simplex):
            self.sort_simplex(simplex, objective)
            centroid = simplex[:-1].mean(axis=0)
            # xr = xo + alpha * (xo - x[n+1])
            reflection = centroid + self.alpha * (simplex[-1] - centroid)
            f0 = objective(simplex[0])
            fn = objective(simplex[-2])
            fw = objective(simplex[-1])
            fr = objective(reflection)
            if f0 <= fr < fn:
                simplex[-1] = reflection
                continue  # goto step 1

            if fr < f0:
                expansion = centroid + self.gamma * (reflection - centroid)
                if objective(expansion) < fr:
                    simplex[-1] = expansion
                else:
                    simplex[-1] = reflection
                continue  # goto step 1

            contraction = centroid + self.rho * (simplex[-1] - centroid)
            if objective(contraction) < fw:
                simplex[-1] = contraction
                continue  # goto step 1

            for i in range(1, n + 1):
                simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])

        f0 = objective(simplex[0])
        fl = objective(simplex[-1])
        if f0 < fl:
            return list(simplex[0]), f0
        else:
            return list(simplex[-1]), fl


N = 4


def main():
    initial_points = [list(random_args_generator(N)) for i in range(2*N - 2)]
    opt = NelderMead()

    result = opt.optimize(F1, initial_points)
    print(result)


if __name__ == '__main__':
    main()
