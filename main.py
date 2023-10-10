##  (-2;9)//	110//	1,25//	4	Скотта	// t-распределение Стьюдента  с числом степеней свободы k=7

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math


## сортировка массива по возрастанию
def sort_massiv(massiv):
    massiv.sort()
    return massiv


def generation_norm_random(a=-2, sigma=3, n_points=110):
    massiv = np.random.normal(a, sigma, n_points)
    return massiv


def plot_norm(a=0, sigma=1, n_points=110):
    massiv = generation_norm_random(a, sigma, n_points)
    plt.hist(massiv, density=True, histtype='stepfilled', alpha=0.8)
    plt.title('Гистограмма нормального распределения')


## расчет числа интервалов  и длину интервала группироваки по правилу Скотта
def calc_k_scott(n_points, massiv):
    h = 3.5 * calc_std(massiv) / (n_points ** (1 / 3))
    k = int((max(massiv) - min(massiv)) / h)
    print('h = ', h)
    print('k = ', k)
    return h, k


## вычисление стандартное отклонение значений ряда измерений в ручную
def calc_std(massiv):
    sum = 0
    for i in massiv:
        sum += i
    mean = sum / len(massiv)
    sum = 0
    for i in massiv:
        sum += (i - mean) ** 2
    return math.sqrt(sum / (len(massiv) - 1))


## вычисление суммы абсолютных частот в интервалах группировки и построить диагармму ассолютных частот
def calc_abs_freq(h, k):
    massiv = np.random.normal(-2, 3, 110)
    ## абсолютные частоты в интервалах группировки
    abs_freq = []
    intervals = []
    for i in range(int(k)):
        intervals.append([massiv.min() + h * i, massiv.min() + h * (i + 1)])
    print('Количество интервалов: ', k)
    print('Длина интервала: ', h)
    print('Интервалы: ', intervals)
    plt.hist(massiv, bins=int(k))
    plt.show()

    return abs_freq


## пункт 4
## функция вычисления математического ожидания
def calc_mean(massiv):
    sum = 0
    for i in massiv:
        sum += i
    return sum / len(massiv)


## функция вычисления дисперсии
def calc_disp(massiv):
    sum = 0
    for i in massiv:
        sum += (i - calc_mean(massiv)) ** 2
    return sum / (len(massiv) - 1)


## функция вычисления среднеквадратического отклонения
def calc_std(massiv):
    return math.sqrt(calc_disp(massiv))


## функция вычисления коэффициента асимметрии
def calc_asym(massiv):
    sum = 0
    for i in massiv:
        sum += (i - calc_mean(massiv)) ** 3
    return sum / (len(massiv) * (calc_std(massiv) ** 3))


## функция вычисления коэффициента эксцесса
def calc_excess(massiv):
    sum = 0
    for i in massiv:
        sum += (i - calc_mean(massiv)) ** 4
    return sum / (len(massiv) * (calc_std(massiv) ** 4)) - 3


## функция вычисления медианы
def calc_median(massiv):
    return massiv[int(len(massiv) / 2)]


def result_4_punkt(massiv):
    print('Математическое ожидание 1 способом: ', calc_mean(massiv))
    print('Математическое ожидание 2 способом: ', np.mean(massiv))
    print('Дисперсия 1 способом: ', calc_disp(massiv))
    print('Дисперсия 2 способом: ', np.var(massiv))
    print('Среднеквадратическое отклонение 1 способом: ', calc_std(massiv))
    print('Среднеквадратическое отклонение 2 способом: ', np.std(massiv))
    print('Коэффициент асимметрии 1 способом: ', calc_asym(massiv))
    print('Коэффициент асимметрии 2 способом: ', sts.skew(massiv))
    print('Коэффициент эксцесса 1 способом: ', calc_excess(massiv))
    print('Коэффициент эксцесса 2 способом: ', sts.kurtosis(massiv))
    print('Медиана 1 способом: ', calc_median(massiv))
    print('Медиана 2 способом: ', np.median(massiv))


massiv = sort_massiv(generation_norm_random(-2, 3, 110))
print(massiv)
calc_abs_freq(calc_k_scott(110, massiv)[0], calc_k_scott(110, massiv)[1])
result_4_punkt(massiv)
