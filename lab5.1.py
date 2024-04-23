import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Definir la función y=3x^3-10x^2-56x+50
x, y = sp.symbols('x y')
f = 3*x**3 - 10*x**2 - 56*x + 50

# Definir derivadas de la función
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f_prime, x)

# Convertir la función a una función lambda para evaluarla numéricamente
f_lambda = sp.lambdify(x, f)

# Definir punto de arranque entre -6 y 6
x_start = np.random.uniform(-6, 6)

# Definir alfa y convergencia
alfa = 0.6
convergencia = 0.001

# Implementar método de Newton-Raphson para dos dimensiones
x_i = x_start
iteraciones = [x_i]

while abs(f_prime.subs(x, x_i)) > convergencia:
    x_i_plus_1 = x_i - alfa * (f_prime.subs(x, x_i) / f_double_prime.subs(x, x_i))
    x_i = x_i_plus_1
    iteraciones.append(x_i)

# Graficar la función y=3x^3-10x^2-56x+50
x_vals = np.linspace(-6, 6, 400)
y_vals = f_lambda(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, label='Función')
plt.scatter(iteraciones, [f_lambda(i) for i in iteraciones], color='red', label='Puntos de iteración')
plt.scatter(x_i, f_lambda(x_i), color='green', label='Mínimo/Máximo encontrado')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gráfica de la función y=3x^3-10x^2-56x+50 con el método de Newton-Raphson para dos dimensiones')
plt.legend()
plt.grid(True)
plt.show()
