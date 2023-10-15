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

    ## Пункт 1


## расчет числа интервалов  и длину интервала группироваки по правилу Скотта
def calc_k_scott(n_points, massiv):
    h = 3.5 * calc_std(massiv) / (n_points ** (1 / 3))
    k = int((max(massiv) - min(massiv)) / h)
    print('h = ', h)
    print('k = ', k)
    return h, k

def interval_grouping(h, k, massiv):
    intervals = []
    for i in range(int(k)):
        intervals.append([massiv.min() + h * i, massiv.min() + h * (i + 1)])
    print('Количество интервалов: ', k)
    print('Длина интервала: ', h)
    print('Интервалы: ', intervals)
    intervals_gist = []
    for i in range(int(k)):
        intervals_gist.append(intervals[i][0])
    return intervals, intervals_gist


## вычисление суммы абсолютных частот в интервалах группировки и построить диагармму ассолютных частот
def calc_abs_freq(h, k, massiv, intervals, intervals_gist):
    ## абсолютные частоты в интервалах группировки
    abs_freq = []
    ## вычисление абсолютных частот
    for i in range(int(k)):
        abs_freq.append(0)
        for j in massiv:
            if intervals[i][0] <= j < intervals[i][1]:
                abs_freq[i] += 1
    print('Абсолютные частоты: ', abs_freq)
    ## сумма абсолютных частот
    sum_abs_freq = 0
    for i in abs_freq:
        sum_abs_freq += i
    print('Сумма абсолютных частот: ', sum_abs_freq)
    ## построение диаграммы абсолютных частот
    plt.bar(intervals_gist, abs_freq, width=h, align='edge')
    plt.title('Диаграмма абсолютных частот')
    plt.show()

def calc_rel_freq(h, k, massiv, intervals, intervals_gist):
    ## относительные частоты в интервалах группировки
    rel_freq = []
## вычисление относительных частот
    for i in range(int(k)):
        rel_freq.append(0)
        for j in massiv:
            if intervals[i][0] <= j < intervals[i][1]:
                rel_freq[i] += 1
        rel_freq[i] /= len(massiv)
    print('Относительные частоты: ', rel_freq)
    ## сумма относительных частот
    sum_rel_freq = 0
    for i in rel_freq:
        sum_rel_freq += i
    print('Сумма относительных частот: ', sum_rel_freq)
    ## построение диаграммы относительных частот
    plt.bar(intervals_gist, rel_freq, width=h, align='edge')
    plt.title('Диаграмма относительных частот')
    plt.show()



    









    ## Пункт 4

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
print(massiv.min())
print(massiv)
calc_abs_freq(calc_k_scott(110, massiv)[0],calc_k_scott(110, massiv)[1], massiv,
              interval_grouping(calc_k_scott(110, massiv)[0],
                                calc_k_scott(110, massiv)[1], massiv)[0], interval_grouping(calc_k_scott(110, massiv)[0],calc_k_scott(110, massiv)[1], massiv)[1])
calc_rel_freq(calc_k_scott(110, massiv)[0],calc_k_scott(110, massiv)[1], massiv,
                interval_grouping(calc_k_scott(110, massiv)[0],
                                calc_k_scott(110, massiv)[1], massiv)[0], interval_grouping(calc_k_scott(110, massiv)[0],calc_k_scott(110, massiv)[1], massiv)[1])


result_4_punkt(massiv)

##Пункт 3
