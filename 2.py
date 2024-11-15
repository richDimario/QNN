import pandas as pd
import numpy as np
import heapq
from collections import defaultdict

# Загрузка данных
adjacency_matrix_path = 'task-2-adjacency_matrix.csv'
nodes_path = 'task-2-nodes.csv'

# Чтение файлов
adjacency_matrix = pd.read_csv(adjacency_matrix_path, index_col=0)
nodes = pd.read_csv(nodes_path)

# Замена значений '-' на бесконечность (отсутствие пути)
adjacency_matrix.replace('-', np.inf, inplace=True)
adjacency_matrix = adjacency_matrix.astype(float)

# Извлечение имен узлов и их весов (вместимость узлов)
node_names = nodes.iloc[:, 0].values
node_capacities = nodes.iloc[:, 1].values

# Преобразование матрицы смежности в numpy массив
adjacency_matrix_np = adjacency_matrix.values

# Проверка размера массивов и исправление несоответствия
if len(node_capacities) < adjacency_matrix_np.shape[0]:
    # Добавляем нулевую вместимость для отсутствующего узла
    node_capacities = np.append(node_capacities, 0)


# Функция поиска кратчайших путей (алгоритм Дейкстры)
def dijkstra_shortest_path(adjacency_matrix, start_node):
    num_nodes = len(adjacency_matrix)
    distances = [np.inf] * num_nodes
    distances[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, travel_time in enumerate(adjacency_matrix[current_node]):
            if travel_time < np.inf:
                distance = current_distance + travel_time
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances


# Функция планирования маршрутов для автобусов
def plan_routes(adjacency_matrix, node_capacities, max_capacity=10, num_buses=15):
    num_nodes = len(adjacency_matrix)
    bus_routes = defaultdict(list)
    bus_loads = [0] * num_buses
    node_visited = [False] * num_nodes

    for node in range(num_nodes):
        if node_capacities[node] > 0:
            assigned = False
            for bus_id in range(num_buses):
                if bus_loads[bus_id] + node_capacities[node] <= max_capacity:
                    bus_routes[bus_id].append(node)
                    bus_loads[bus_id] += node_capacities[node]
                    node_visited[node] = True
                    assigned = True
                    break

            if not assigned:
                remaining_capacity = node_capacities[node]
                for bus_id in range(num_buses):
                    if bus_loads[bus_id] < max_capacity:
                        possible_load = min(max_capacity - bus_loads[bus_id], remaining_capacity)
                        bus_routes[bus_id].append(node)
                        bus_loads[bus_id] += possible_load
                        remaining_capacity -= possible_load
                        if remaining_capacity <= 0:
                            node_visited[node] = True
                            break

    return bus_routes, bus_loads


# Выполнение планирования маршрутов
bus_routes, bus_loads = plan_routes(adjacency_matrix_np, node_capacities)

# Вывод результатов
print("Маршруты автобусов:")
for bus_id, route in bus_routes.items():
    print(f"Автобус {bus_id + 1}: Узлы {route}")

print("\nЗагрузки автобусов:")
for bus_id, load in enumerate(bus_loads):
    print(f"Автобус {bus_id + 1}: {load} человек")
