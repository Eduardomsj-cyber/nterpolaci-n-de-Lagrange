import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import fsolve

def f(x):
    return np.sin(x) - x/2

# Selección de tres puntos equidistantes en el intervalo [0,2]
x_points = np.linspace(0, 2, 3)
y_points = f(x_points)

# Construcción del polinomio interpolante de Lagrange
poly = lagrange(x_points, y_points)

# Encontrar la raíz del polinomio interpolante
root_interp = fsolve(poly, 1)[0]  # Se inicia la búsqueda en x=1

# Cálculo de errores
root_real = fsolve(f, 1)[0]  # Raíz real de f(x)
error_abs = abs(root_real - root_interp)
error_rel = error_abs / abs(root_real)
error_quad = error_abs**2

# Impresión de resultados
print(f"Raíz interpolada: {root_interp:.6f}")
print(f"Raíz real: {root_real:.6f}")
print(f"Error absoluto: {error_abs:.6e}")
print(f"Error relativo: {error_rel:.6e}")
print(f"Error cuadrático: {error_quad:.6e}")

# Generación de gráficos
x_vals = np.linspace(0, 2, 100)
y_vals = f(x_vals)
y_interp_vals = poly(x_vals)

plt.figure(figsize=(10,5))
plt.plot(x_vals, y_vals, label='f(x) = sin(x) - x/2', color='blue')
plt.plot(x_vals, y_interp_vals, label='Polinomio interpolante', linestyle='--', color='red')
plt.scatter(x_points, y_points, color='black', label='Puntos de interpolación')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root_interp, color='green', linestyle=':', label=f'Raíz interpolada: {root_interp:.6f}')
plt.axvline(root_real, color='purple', linestyle='--', label=f'Raíz real: {root_real:.6f}')
plt.legend()
plt.title("Interpolación de Lagrange y convergencia de la raíz")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()
