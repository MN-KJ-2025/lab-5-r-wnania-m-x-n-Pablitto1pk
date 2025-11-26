import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def spare_matrix_Abt(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja tworząca zestaw składający się z macierzy A (m,n) i
    wektora b (m,) na podstawie pomocniczego wektora t (m,).

    Args:
        m (int): Liczba wierszy macierzy A.
        n (int): Liczba kolumn macierzy A.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A o rozmiarze (m,n),
            - Wektor b (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    if not isinstance(m, int) or not isinstance(n, int):
        return None
    if m <= 0 or n <= 0:
        return None

    t = np.linspace(0, 1, m)

    A = np.zeros((m, n), dtype=float)
    for i in range(m):
        A[i] = t[i] ** np.arange(n)

    b = np.cos(4 * t)

    return A, b


def square_from_rectan(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników
    na kwadratowy układ równań.
    A^T * A * x = A^T * b  ->  A_new * x = b_new

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A_new o rozmiarze (n,n),
            - Wektor b_new (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    if not isinstance(A, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if A.ndim != 2 or b.ndim != 1:
        return None
    if A.shape[0] != b.shape[0]:
        return None

    try:
        A_new = A.T @ A
        b_new = A.T @ b
    except Exception:
        return None

    return A_new, b_new


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if A.ndim != 2 or b.ndim != 1 or x.ndim != 1:
        return None

    m, n = A.shape
    if x.shape[0] != n or b.shape[0] != m:
        return None

    try:
        r = b - A @ x
    except Exception:
        return None

    return float(np.linalg.norm(r))
'''
    ##Macierz A : [[2 3 3 3] [9 8 3 0] [2 2 4 9] [4 6 1 9] [8 2 0 7] [4 8 4 0]] Wektor b : [3 4 8 1 6 2]
A = np.array([
    [2, 3, 3, 3],
    [9, 8, 3, 0],
    [2, 2, 4, 9],
    [4, 6, 1, 9],
    [8, 2, 0, 7],
    [4, 8, 4, 0]
], dtype=float)

b = np.array([3, 4, 8, 1, 6, 2], dtype=float)

Q, R = np.linalg.qr(A, mode="reduced")
y = Q.T @ b
x = sp.linalg.solve_triangular(R, y, lower=False)
# Trzecia współrzędna
x3 = x[2]

print(x3)
'''