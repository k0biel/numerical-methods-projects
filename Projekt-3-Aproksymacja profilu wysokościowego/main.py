import pandas as pd
import matplotlib.pyplot as plt
import math

def lagrange_interpolation(x, y, x_new):
    n = len(x)
    y_new = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (x_new - x[j]) / (x[i] - x[j])
        y_new += y[i] * p
    return y_new

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

def splines_interpolation(x, y, x_new):
    n = len(x)
    h = [x[i+1] - x[i] for i in range(n-1)]
    A = [[0] * n for _ in range(n)]
    b = [[0] for _ in range(n)]
    A[0][0] = 1
    A[n - 1][n - 1] = 1
    for i in range(1, n-1):
        A[i][i-1] = h[i-1]
        A[i][i] = 2 * (h[i-1] + h[i])
        A[i][i+1] = h[i]
        b[i][0] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])

    c, _, _ = jacobi(A, b)
    a = [y[i] for i in range(n - 1)]
    b = [(y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i][0] + c[i + 1][0]) / 3 for i in range(n - 1)]
    d = [(c[i + 1][0] - c[i][0]) / (3 * h[i]) for i in range(n - 1)]
    c = [c[i][0] for i in range(n)]

    y_new = []
    for x_val in x_new:
        for i in range(n - 1):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                y_val = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
                y_new.append(y_val)
                break
        else:
            y_new.append(None)
    return y_new

def generate_linear_indices(start_index, end_index, n):
    distance = (end_index - start_index) / (n - 1)
    points = [int(start_index + i * distance) for i in range(n)]
    return points

def generate_chebyshev_indices(start_index, end_index, n):
    cos_values = [math.cos((2 * i - 1) * math.pi / (2 * n)) for i in range(1, n + 1)]
    points = [0.5 * (end_index - start_index) * x + 0.5 * (start_index + end_index) for x in cos_values]
    return [round(point) for point in points]

def interpolate(x, y, method='lagrange', node_distribution=None, num_nodes=None):
    if node_distribution == 'chebyshev':
        indices = generate_chebyshev_indices(0, len(x) - 1, num_nodes)
        indices.sort()
    elif node_distribution == 'linear':
        indices = generate_linear_indices(0, len(x) - 1, num_nodes)

    x_nodes = [x[i] for i in indices]
    y_nodes = [y[i] for i in indices]

    if method == 'splines':
        return x, x_nodes, y_nodes, splines_interpolation(x_nodes, y_nodes, x)
    elif method == 'lagrange':
        return x, x_nodes, y_nodes, [lagrange_interpolation(x_nodes, y_nodes, xi) for xi in x]

data = pd.read_csv('SpacerniakGdansk.csv', header=None, skiprows=1)
x = data[0].tolist()
y = data[1].tolist()

method = 'splines'  # wybór 'lagrange' lub 'splines'
node_distribution = 'chebyshev'  # wybór 'linear' lub 'chebyshev'
num_nodes = 50 # wybór liczby węzłów interpolacji

if method == 'lagrange':
    label = f"Interpolacja Lagrange'a na {num_nodes} węzłach"
elif method == 'splines':
    label = f"Interpolacja funkcjami sklejanymi trzeciego stopnia na {num_nodes} węzłach"

x_new, x_nodes, y_nodes, y_new = interpolate(x, y, method=method, node_distribution=node_distribution, num_nodes=num_nodes)

plt.figure(figsize=(10, 6))
plt.title(label)
plt.xlabel('Dystans [m]')
plt.ylabel('Wysokość [m]')
plt.plot(x, y, label='Dane oryginalne')
plt.plot(x_new, y_new, color='black', label='Interpolacja')
plt.scatter(x_nodes, y_nodes, color='red', label='Węzły interpolacji')
plt.grid(True)
plt.legend()
plt.show()