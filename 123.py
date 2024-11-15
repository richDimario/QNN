import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

# Количество акций
num_assets = len(expected_returns)

# Функция для расчёта доходности и риска портфеля
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Целевая функция (максимизация доходности)
def objective_function(weights):
    return -np.dot(weights, expected_returns)

# Ограничения: сумма весов = 1 и ограничение по риску
constraints = (
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
    {'type': 'ineq', 'fun': lambda weights: target_risk - portfolio_performance(weights, expected_returns, cov_matrix)[1]}
)

# Границы для весов (от 0 до 1)
bounds = tuple((0, 1) for _ in range(num_assets))

# Начальное предположение (равномерное распределение)
initial_guess = np.array(num_assets * [1. / num_assets])

# Оптимизация
optimized_result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Оптимальные веса
optimized_weights = optimized_result.x

# Расчёт доходности и риска оптимального портфеля
opt_return, opt_risk = portfolio_performance(optimized_weights, expected_returns, cov_matrix)

# Формирование DataFrame с результатами
portfolio_df = pd.DataFrame({
    'Stock': data.columns,
    'Weight': optimized_weights
})

# Вывод результатов
print(portfolio_df)
print(f'\nОжидаемая доходность портфеля: {opt_return * 100:.2f}%')
print(f'Риск портфеля (стандартное отклонение): {opt_risk * 100:.2f}%')
