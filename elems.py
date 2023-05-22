import numpy as np
import matplotlib.pyplot as plt

# Параметры модели и сетки
D = 0.1  # Коэффициент диффузии
L = 1.0  # Длина области
T = 1.0  # Общее время
N = 200  # Количество узлов в пространственной сетке
M = 1000 # Количество узлов во временной сетке
dx = L / (N - 1)  # Шаг по пространству
dt = T / (M - 1)  # Шаг по времени

# Создание сетки
x = np.linspace(0, L, N)
t = np.linspace(0, T, M)

# Создание матрицы для численного решения
u = np.zeros((N, M))

# Задание начального условия
u[:, 0] = np.sin(np.pi * x / L)

# Матрица жесткости
K = np.zeros((N, N))

for i in range(1, N - 1):
    K[i, i] = 2 * D / dx**2
    K[i, i-1] = -D / dx**2
    K[i, i+1] = -D / dx**2

K[0, 0] = 1
K[N-1, N-1] = 1

# Численное решение
for j in range(M - 1):
    F = np.zeros(N)
    F[0] = u[0, j]
    F[-1] = u[-1, j]

    for i in range(1, N - 1):
        F[i] = u[i, j]

    u[:, j + 1] = np.linalg.solve(K, F)

# Визуализация результатов
plt.figure()
for j in range(M):
    plt.plot(x, u[:, j], label=f"t = {t[j]:.2f}")
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
