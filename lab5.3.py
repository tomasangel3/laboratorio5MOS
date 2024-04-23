import sympy as sp
import matplotlib.pyplot as plt

# Definir variables simbólicas
x, y = sp.symbols('x y')

# Definir función z = ((x - 1)^2) + 100*(y - x^2)^2
z = (x - 1)**2 + 100*(y - x**2)**2

# Calcular gradiente y matriz hessiana
gradiente = [sp.diff(z, var) for var in (x, y)]
hessiana = [[sp.diff(g, var) for var in (x, y)] for g in gradiente]

# Convertir la función a una función lambda para evaluarla numéricamente
z_lambda = sp.lambdify((x, y), z)

# Definir punto de inicio
x_start = sp.Matrix([0, 10])

# Inicializar lista para almacenar puntos de iteración
iteraciones = [x_start]

# Definir alfa y convergencia
alfa = 0.001
convergencia = 0.001

# Implementar método de Newton-Raphson para tres dimensiones
while sp.sqrt(sum([g.subs({x: x_start[0], y: x_start[1]}).evalf()**2 for g in gradiente])) > convergencia:
    hessiana_inv = sp.Matrix(hessiana).inv()
    gradiente_val = sp.Matrix([g.subs({x: x_start[0], y: x_start[1]}).evalf() for g in gradiente])
    x_start = x_start - alfa * hessiana_inv * gradiente_val
    iteraciones.append(x_start)

# Graficar la función z = ((x - 1)^2) + 100*(y - x^2)^2
x_vals = sp.linspace(-2, 2, 400)
y_vals = sp.linspace(-1, 15, 400)
X, Y = sp.meshgrid(x_vals, y_vals)
Z = z_lambda(X, Y)

plt.figure(figsize=(12, 6))
plt.contour(X, Y, Z, levels=sp.logspace(-1, 3, 10))
plt.scatter(*zip(*[(float(p[0]), float(p[1])) for p in iteraciones]), color='cyan', label='Ruta de puntos encontrados')
plt.scatter(*zip(*[(float(iteraciones[-1][0]), float(iteraciones[-1][1]))]), color='red', label='Mínimo encontrado')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Función z = ((x - 1)^2) + 100*(y - x^2)^2 con método de Newton-Raphson')
plt.legend()
plt.show()
