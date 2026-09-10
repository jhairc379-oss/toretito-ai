import torch
import torch.nn as nn

# Datos (normalizados, igual que antes)
tamanos = [50, 60, 70, 80, 90, 100]
habitaciones = [1, 2, 2, 3, 3, 4]
precios = [160, 195, 225, 265, 295, 335]

# PyTorch trabaja con "tensores" (son como listas, pero optimizadas)
# Cada fila es una casa: [tamaño_normalizado, habitaciones_normalizado]
X = torch.tensor([[t/100, h/10] for t, h in zip(tamanos, habitaciones)], dtype=torch.float32)
y = torch.tensor([[p/100] for p in precios], dtype=torch.float32)

print("Datos de entrada (X):")
print(X)
print("\nDatos reales (y):")
print(y)
# Definimos la red: 2 entradas -> 3 neuronas ocultas (con ReLU) -> 1 salida
modelo = nn.Sequential(
    nn.Linear(2, 3),   # capa oculta: recibe 2 entradas, tiene 3 neuronas
    nn.ReLU(),         # función de activación
    nn.Linear(3, 1)    # capa de salida: recibe 3, da 1 número (el precio)
)

print(modelo)
# El "optimizador" es quien ajusta los pesos (equivalente a tu tasa de aprendizaje manual)
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01)

# La función de pérdida mide el error (equivalente a tu "error = (prediccion - real)^2")
funcion_perdida = nn.MSELoss()

epocas = 3000

for epoca in range(epocas):
    # 1. Forward: calcular predicciones
    predicciones = modelo(X)

    # 2. Calcular el error
    perdida = funcion_perdida(predicciones, y)

    # 3. Backward: PyTorch calcula TODAS las derivadas automáticamente
    optimizador.zero_grad()  # limpiar gradientes anteriores
    perdida.backward()       # aquí ocurre todo el backpropagation, solo!

    # 4. Ajustar los pesos usando esos gradientes
    optimizador.step()

    if epoca % 300 == 0:
        print(f"Época {epoca} | Pérdida: {perdida.item():.4f}")

print("\nEntrenamiento terminado.\n")

# Predicciones finales (multiplicamos por 100 para volver a la escala real)
with torch.no_grad():
    predicciones_finales = modelo(X) * 100

print("Predicciones finales:")
for i in range(len(tamanos)):
    print(f"Tamaño: {tamanos[i]}m2, {habitaciones[i]} hab -> Predicción: {predicciones_finales[i].item():.1f} | Real: {precios[i]}")