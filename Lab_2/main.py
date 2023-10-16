import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import statistics
import scipy.stats as sts

n = 16
a = -1
gamma = 0.9
sigma = 4
M = 1000

sample = np.random.normal(a, sigma, n)  # создание выборки
mean = statistics.mean(sample)

t2 = (1 + gamma)/2
t = sts.norm().ppf(t2)
print(t)

delta = t*sigma / np.sqrt(n)
I_left = mean-delta
I_right = mean + delta
print(I_left,I_right)



