"""

Одномерная оптимизация.
Написать программу для решения следующей задачи:
Минимизировать функцию f(x) на отрезке a <= x <= b

1. f(x) = x, 0<=x<=10
2. f(x) = (x - 3)^2+4, 0<=x<=10
3. f(x) = ((x - 3)^2 - 4)^2, 0<=x<=10
4. f(x) = ((x - 3)^2 - 4)^2 + 12x, 0<=x<=10

Методы:
1. Перебора по x_i = a + i/n*(b-a), 0<=i<=n
2. Метод дихотомии с фиксированным delta = 10^(-9)
3. Метод золотого сечения
4. Метод Ньютона (знание производной)

"""


class Function:
    def __init__(self, function):
        self.counter = 0
        self.f = function

    def __call__(self, x):
        self.counter += 1
        return self.f(x)


class Optimizer:
    def __init__(self):
        pass

    def optimize(self, function, a, b, derivative=None):
        """Здесь пишем код метода оптимизации"""
        x_opt = a
        f_opt = function(x_opt)
        return x_opt, f_opt


def f1(x):
    return x


def main():
    optimizer = Optimizer()
    function = Function(f1)
    x, f = optimizer.optimize(function, 0, 10)
    print(f'Минимальное значение {f} функции f1(x) достигается в точке x = {x}')
    print(f'Функция f1 была вызвана {function.count} раз')


if __name__ == '__main__':
    main()
