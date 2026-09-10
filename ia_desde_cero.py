# Datos de entrenamiento: tamaño (m2) y precio real (miles de USD)
tamanos = [50, 60, 70, 80, 90, 100]
precios = [150, 180, 210, 240, 270, 300]
# Valores iniciales (al azar, la IA no sabe nada todavía)
peso = 0.1
bias = 0.0

def predecir(tamano):
    return tamano * peso + bias

# Probemos la predicción ANTES de entrenar
print("Predicciones iniciales (sin entrenar):")
for t, p_real in zip(tamanos, precios):
    prediccion = predecir(t)
    print(f"Tamaño: {t}m2 -> Predicción: {prediccion:.1f} | Real: {p_real}")
    # --- ENTRENAMIENTO ---
tasa_aprendizaje = 0.0001  # qué tan grande es cada ajuste
epocas = 1000  # cuántas veces repetimos el proceso

for epoca in range(epocas):
    error_total = 0

    for t, p_real in zip(tamanos, precios):
        prediccion = predecir(t)
        error = prediccion - p_real

        # Ajustamos peso y bias en la dirección que reduce el error
        peso = peso - tasa_aprendizaje * error * t
        bias = bias - tasa_aprendizaje * error

        error_total += error ** 2

    # Cada 100 épocas, mostramos cómo va el aprendizaje
    if epoca % 100 == 0:
        print(f"Época {epoca} | Error total: {error_total:.2f} | peso: {peso:.4f} | bias: {bias:.4f}")

print("\nEntrenamiento terminado.")
print(f"Peso final: {peso:.4f} | Bias final: {bias:.4f}\n")

# Probemos de nuevo, ya entrenada
print("Predicciones DESPUÉS de entrenar:")
for t, p_real in zip(tamanos, precios):
    prediccion = predecir(t)
    print(f"Tamaño: {t}m2 -> Predicción: {prediccion:.1f} | Real: {p_real}")
    # Probemos con un tamaño que NO estaba en los datos de entrenamiento
nuevo_tamano = 75
prediccion_nueva = predecir(nuevo_tamano)
print(f"\nPredicción para una casa de {nuevo_tamano}m2 (nunca vista): {prediccion_nueva:.1f}")