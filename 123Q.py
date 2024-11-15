import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dimod import BinaryQuadraticModel
from dwave.system import LeapHybridSampler

# Загрузка данных
data = pd.read_csv('task-1-stocks.csv')

# Расчёт дневных доходностей
returns = data.pct_change().dropna()

# Расчёт ковариационной матрицы доходностей
cov_matrix = returns.cov()

# Ожидаемые доходности (средние доходности акций)
expected_returns = returns.mean()

# Целевой уровень риска
target_risk = 0.2
num_assets = len(expected_returns)
num_bits = 4  # Количество битов для кодирования одного веса

# Генерация бинарных переменных
Q = {}

# Формирование QUBO модели
for i in range(num_assets):
    for j in range(num_assets):
        for k in range(num_bits):
            for l in range(num_bits):
                weight_i = 2**(-k)
                weight_j = 2**(-l)
                # Целевая функция: максимизация доходности
                Q[(i * num_bits + k, j * num_bits + l)] = (
                    -weight_i * weight_j * cov_matrix.iloc[i, j]
                    + weight_i * expected_returns[i] * (i == j)
                )

# Добавление ограничений (сумма весов = 1)
for i in range(num_assets):
    for k in range(num_bits):
        weight_i = 2**(-k)
        Q[(i * num_bits + k, i * num_bits + k)] += 100 * (weight_i**2)

# Решение задачи с использованием LeapHybridSampler
bqm = BinaryQuadraticModel.from_qubo(Q)
sampler = LeapHybridSampler()
solution = sampler.sample(bqm).first

# Преобразование решения обратно в веса
optimized_weights = []
for i in range(num_assets):
    weight = sum(solution.sample[i * num_bits + k] * 2**(-k) for k in range(num_bits))
    optimized_weights.append(weight)

# Нормализация весов
optimized_weights = np.array(optimized_weights)
optimized_weights /= np.sum(optimized_weights)

# Расчёт доходности и риска оптимального портфеля
opt_return = np.dot(optimized_weights, expected_returns)
opt_risk = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))

# Формирование DataFrame с результатами
portfolio_df = pd.DataFrame({
    'Stock': data.columns,
    'Weight': optimized_weights
})

# Вывод результатов
print(portfolio_df)
print(f'\nОжидаемая доходность портфеля: {opt_return * 100:.2f}%')
print(f'Риск портфеля (стандартное отклонение): {opt_risk * 100:.2f}%')
