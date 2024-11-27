import math
import time
import matplotlib.pyplot as plt
import copy

# Indeks - 193618
c = 1
d = 8
e = 6
f = 3
N = 9 * 100 + 1 * 10 + 8
a2 = a3 = -1

# Inicjalizacja macierzy A
def create_matrix(N, a1, a2, a3):
    A = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        A[i][i] = a1
        if i < N - 1:
            A[i][i + 1] = a2
            A[i + 1][i] = a2
        if i < N - 2:
            A[i][i + 2] = a3
            A[i + 2][i] = a3
    return A

# Inicjalizacja wektora b
def create_vector(N, f):
    b = [[0] for _ in range(N)]
    for i in range(N):
        b[i][0] = math.sin(i * (f + 1))
    return b

def matrix_sub(A, B):
    N = len(A)
    M = len(B[0])
    result = [[0 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            result[i][j] = A[i][j] - B[i][j]
    return result

def matrix_multiply(A, B):
    N = len(A)
    M = len(B[0])
    result = [[0 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def residual(A, b, x):
    return matrix_sub(matrix_multiply(A, x), b)


def norm(residual):
    return math.sqrt(sum(x[0] ** 2 for x in residual))


def jacobi(A, b):
    iterations = 0
    n = len(A)
    x = [[0] for _ in range(n)]
    residuum_norms = []

    while True:
        x_new = [[0] for _ in range(n)]

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j][0]
            x_new[i][0] = (b[i][0] - sigma) / A[i][i]

        residuum_norm = norm(residual(A, b, x_new))
        residuum_norms.append(residuum_norm)

        x = x_new
        iterations += 1

        if residuum_norm <= 10 ** -9 or iterations >= 100:
            break

    return x, iterations, residuum_norms

def gauss_seidel(A, b):
    iterations = 0
    n = len(A)
    x = [[0] for _ in range(n)]
    residuum_norms = []

    while True:
        x_new = [[0] for _ in range(n)]

        for i in range(n):
            sigma = 0
            for j in range(i):
                sigma += A[i][j] * x_new[j][0]
            for j in range(i + 1, n):
                sigma += A[i][j] * x[j][0]
            x_new[i][0] = (b[i][0] - sigma) / A[i][i]

        residuum_norm = norm(residual(A, b, x_new))
        residuum_norms.append(residuum_norm)

        x = x_new
        iterations += 1

        if residuum_norm <= 10 ** -9 or iterations >= 100:
            break

    return x, iterations, residuum_norms

def lu_decomposition(A):
    m = len(A)
    U = copy.deepcopy(A)
    L = [[0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        L[i][i] = 1
    for i in range(1, m):
        for j in range(i):
            L[i][j] = U[i][j] / U[j][j]
            for k in range(m):
                U[i][k] = U[i][k] - L[i][j] * U[j][k]
    return L, U

def forward_substitution(L, b):
    n = len(L)
    y = [[0] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            y[i][0] -= L[i][j] * y[j][0]
        y[i][0] += b[i][0]
    return y

def backward_substitution(U, y):
    n = len(U)
    x = [[0] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i][0] -= U[i][j] * x[j][0]
        x[i][0] += y[i][0]
        x[i][0] /= U[i][i]
    return x

def lu_solve(A, b):
    start = time.time()
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    residuum_norm = norm(residual(A, b, x))
    end = time.time()
    print("Norma residuum dla metody faktoryzacji LU wynosi:", residuum_norm)
    print(f"Czas trwania algorytmu faktoryzacji LU: {end - start} sekund")
    return x

def plot_test(A, b):
    start = time.time()
    x_jacobi, iterations_jacobi, residuals_jacobi = jacobi(A, b)
    end = time.time()
    print(f"Czas trwania algorytmu Jacobiego: {end - start} sekund")
    print(f"Liczba iteracji algorytmu Jacobiego: {iterations_jacobi}")

    start = time.time()
    x_gauss, iterations_gauss, residuals_gauss = gauss_seidel(A, b)
    end = time.time()
    print(f"Czas trwania algorytmu Gaussa-Seidla: {end - start} sekund")
    print(f"Liczba iteracji algorytmu Gaussa-Seidla: {iterations_gauss}")

    plt.figure(figsize=(12, 6))
    plt.plot(residuals_jacobi, label='Jacobi')
    plt.plot(residuals_gauss, label='Gauss-Seidel', color = 'red')
    plt.title('Jak zmienia się norma residuum w kolejnych iteracjach?')
    plt.xlabel('Liczba iteracji')
    plt.ylabel('Norma residuum')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_execution_times(N_values):
    times_jacobi = []
    times_gauss = []
    times_lu = []

    for N in N_values:
        A = create_matrix(N, 5 + e, a2, a3)
        b = create_vector(N, f)

        start = time.time()
        jacobi(A, b)
        end = time.time()
        times_jacobi.append(end - start)

        start = time.time()
        gauss_seidel(A, b)
        end = time.time()
        times_gauss.append(end - start)

        start = time.time()
        lu_solve(A, b)
        end = time.time()
        times_lu.append(end - start)

    plt.figure(figsize=(12, 6))
    plt.plot(N_values, times_jacobi, label='Jacobi')
    plt.plot(N_values, times_gauss, label='Gauss-Seidel', color='red')
    plt.plot(N_values, times_lu, label='LU', color='green')
    plt.title('Zależności czasu wyznaczenia rozwiązania dla trzech badanych metod w zależności od liczby niewiadomych')
    plt.xlabel('Liczba niewiadomych N')
    plt.ylabel('Czas [s]')
    plt.grid(True)
    plt.legend()
    plt.show()


# Zadanie A
A = create_matrix(N, 5 + e, a2, a3)
b = create_vector(N, f)

# Zadanie B
plot_test(A, b)

# Zadanie C
# A = create_matrix(N, 3, a2, a3)
# plot_test(A, b)

# Zadanie D
# x_lu = lu_solve(A, b)

# Zadanie E
# N_values = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
# plot_execution_times(N_values)
