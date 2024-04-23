import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Definir la función y=x^5-8x^3+10x+6
x = sp.symbols('x')
f = x**5 - 8*x**3 + 10*x + 6

# Definir derivadas de la función
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

# Convertir la función a una función lambda para evaluarla numéricamente
f_lambda = sp.lambdify(x, f)

# Definir intervalo y puntos de inicio
intervalo = np.linspace(-3, 3, 1000)
puntos_inicio = [-3, -2, -1, 0, 1, 2, 3]

# Inicializar listas para almacenar mínimos y máximos locales
minimos_locales = []
maximos_locales = []

# Iterar sobre los puntos de inicio y encontrar mínimos y máximos locales
for punto_inicio in puntos_inicio:
    x_i = punto_inicio
    iteraciones = [x_i]
    while True:
        f_prime_value = sp.re(f_prime.subs(x, x_i))
        if not f_prime_value.is_number:
            break
        if abs(f_prime_value) < 0.001:
            break
        x_i_plus_1 = x_i - 0.6 * (f_prime_value / f_double_prime.subs(x, x_i))
        x_i = x_i_plus_1
        iteraciones.append(x_i)
    extremo = (x_i, f_lambda(x_i))
    if f_double_prime.subs(x, x_i) > 0:
        minimos_locales.append(extremo)
    elif f_double_prime.subs(x, x_i) < 0:
        maximos_locales.append(extremo)

# Encontrar mínimo y máximo global
extremos_totales = minimos_locales + maximos_locales
minimo_global = min(extremos_totales, key=lambda x: x[1])
maximo_global = max(extremos_totales, key=lambda x: x[1])

# Graficar la función y=x^5-8x^3+10x+6
y_vals = f_lambda(intervalo)

plt.figure(figsize=(12, 6))
plt.plot(intervalo, y_vals, label='Función')

# Graficar mínimos y máximos locales en negro
for extremo in minimos_locales:
    plt.scatter(extremo[0], extremo[1], color='black')
for extremo in maximos_locales:
    plt.scatter(extremo[0], extremo[1], color='black')

# Graficar mínimo y máximo global en rojo
plt.scatter(minimo_global[0], minimo_global[1], color='red', label='Mínimo global')
plt.scatter(maximo_global[0], maximo_global[1], color='red', label='Máximo global')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfica de la función y=x^5-8x^3+10x+6 con todos los mínimos y máximos')
plt.legend()
plt.grid(True)
plt.show()
