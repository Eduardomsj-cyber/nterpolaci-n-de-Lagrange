import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import fsolve

def funcion(x):
    return x**3 - 6*x**2 + 11*x - 6

def error_cuadratico(real, estimado):
    return np.mean((real - estimado) ** 2)

# Paso 1: Selección de puntos adecuados
puntos_x = np.array([1, 2, 3])
puntos_y = funcion(puntos_x)

# Paso 2: Construcción del polinomio interpolante
polinomio = lagrange(puntos_x, puntos_y)

# Paso 3: Encontrar la raíz del polinomio interpolante
raiz_estimada = fsolve(polinomio, 1.5)[0]
raiz_real = 2  # La raíz exacta en este intervalo

# Paso 4: Gráficos
x_vals = np.linspace(0.5, 3.5, 400)
y_vals_func = funcion(x_vals)
y_vals_polinomio = polinomio(x_vals)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals_func, label='Función original f(x)', linestyle='dashed')
plt.plot(x_vals, y_vals_polinomio, label='Polinomio interpolante P(x)')
plt.scatter(puntos_x, puntos_y, color='red', label='Puntos de interpolación')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(raiz_estimada, color='green', linestyle='dotted', label=f'Raíz estimada: {raiz_estimada:.4f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolación de Lagrange')
plt.grid()
plt.show()

# Paso 5: Cálculo de errores
error_absoluto = abs(raiz_real - raiz_estimada)
error_relativo = error_absoluto / abs(raiz_real)
error_cuadratico_valor = error_cuadratico(np.array([raiz_real]), np.array([raiz_estimada]))

# Imprimir resultados
data = [['Error Absoluto', error_absoluto],
        ['Error Relativo', error_relativo],
        ['Error Cuadrático', error_cuadratico_valor]]

print("Resultados del cálculo de errores:")
for row in data:
    print(f"{row[0]}: {row[1]:.6f}")
