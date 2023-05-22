import numpy as np
import matplotlib.pyplot as plt

# Параметры модели и сетки
D = 0.1  # Коэффициент диффузии
L = 1.0  # Длина области
T = 1.0  # Общее время
N = 100  # Количество узлов в пространственной сетке
M = 1000  # Количество узлов во временной сетке
dx = L / (N - 1)  # Шаг по пространству
dt = T / (M - 1)  # Шаг по времени

# Создание сетки
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

# Создание матрицы для численного решения
u = np.zeros((N, M))

# Задание начального условия
u[:, 0] = np.sin(np.pi * x / L)

# Численное решение
for j in range(M - 1):
    for i in range(1, N - 1):
        u[i, j + 1] = u[i, j] + D * dt / dx**2 * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) - dt / dx * (D * np.log(u[i, j]) * (u[i + 1, j] - u[i - 1, j])) / (2 * dx)

# Визуализация результатов
plt.figure()
for j in range(M):
    plt.plot(x, u[:, j], label=f"t = {t[j]:.2f}")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
