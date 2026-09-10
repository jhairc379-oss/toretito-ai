# Datos: tamaño (m2), habitaciones, precio real (miles de USD)
tamanos = [50, 60, 70, 80, 90, 100]
habitaciones = [1, 2, 2, 3, 3, 4]
precios = [160, 195, 225, 265, 295, 335]
# Valores iniciales (al azar)
peso1 = 0.1  # importancia del tamaño
peso2 = 0.1  # importancia de las habitaciones
bias = 0.0

def predecir(tamano, hab):
    return tamano * peso1 + hab * peso2 + bias

print("Predicciones iniciales (sin entrenar):")
for t, h, p_real in zip(tamanos, habitaciones, precios):
    prediccion = predecir(t, h)
    print(f"Tamaño: {t}m2, {h} hab -> Predicción: {prediccion:.1f} | Real: {p_real}")
    # --- ENTRENAMIENTO ---
tasa_aprendizaje = 0.0001
epocas = 20000

for epoca in range(epocas):
    error_total = 0

    for t, h, p_real in zip(tamanos, habitaciones, precios):
        prediccion = predecir(t, h)
        error = prediccion - p_real

        # Ajustamos cada peso según cuánto influyó su entrada en el error
        peso1 = peso1 - tasa_aprendizaje * error * t
        peso2 = peso2 - tasa_aprendizaje * error * h
        bias = bias - tasa_aprendizaje * error

        error_total += error ** 2

    if epoca % 100 == 0:
        print(f"Época {epoca} | Error total: {error_total:.2f} | peso1: {peso1:.4f} | peso2: {peso2:.4f} | bias: {bias:.4f}")

print("\nEntrenamiento terminado.")
print(f"peso1 (tamaño): {peso1:.4f} | peso2 (habitaciones): {peso2:.4f} | bias: {bias:.4f}\n")

print("Predicciones DESPUÉS de entrenar:")
for t, h, p_real in zip(tamanos, habitaciones, precios):
    prediccion = predecir(t, h)
    print(f"Tamaño: {t}m2, {h} hab -> Predicción: {prediccion:.1f} | Real: {p_real}")