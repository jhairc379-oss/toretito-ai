def relu(x):
    return x if x > 0 else 0

def relu_derivada(x):
    return 1 if x > 0 else 0

# Datos originales
tamanos_orig = [50, 60, 70, 80, 90, 100]
habitaciones_orig = [1, 2, 2, 3, 3, 4]
precios_orig = [160, 195, 225, 265, 295, 335]

# Normalizamos dividiendo por un número grande, para que queden pequeños
tamanos = [t / 100 for t in tamanos_orig]
habitaciones = [h / 10 for h in habitaciones_orig]
precios = [p / 100 for p in precios_orig]

# Capa oculta: 3 neuronas
n1 = {"pt": 0.05, "ph": 0.3, "b": 0.0}
n2 = {"pt": -0.02, "ph": 0.1, "b": 0.5}
n3 = {"pt": 0.08, "ph": -0.2, "b": -1.0}

# Neurona de salida
salida = {"p1": 1.0, "p2": 1.0, "p3": 1.0, "b": 0.0}

tasa = 0.001
epocas = 5000

for epoca in range(epocas):
    error_total = 0

    for t, h, real in zip(tamanos, habitaciones, precios):
        # --- Adelante (forward pass) ---
        z1 = t * n1["pt"] + h * n1["ph"] + n1["b"]
        z2 = t * n2["pt"] + h * n2["ph"] + n2["b"]
        z3 = t * n3["pt"] + h * n3["ph"] + n3["b"]

        s1, s2, s3 = relu(z1), relu(z2), relu(z3)

        prediccion = s1 * salida["p1"] + s2 * salida["p2"] + s3 * salida["p3"] + salida["b"]

        error = prediccion - real
        error_total += error ** 2

        # --- Atrás (backpropagation) ---
        d_error_pred = 2 * error  # eslabón 1

        # Ajustar neurona de salida
        salida["p1"] -= tasa * d_error_pred * s1
        salida["p2"] -= tasa * d_error_pred * s2
        salida["p3"] -= tasa * d_error_pred * s3
        salida["b"]  -= tasa * d_error_pred

        # Ajustar neurona 1 (propagando el error hacia atrás)
        d_n1 = d_error_pred * salida["p1"] * relu_derivada(z1)
        n1["pt"] -= tasa * d_n1 * t
        n1["ph"] -= tasa * d_n1 * h
        n1["b"]  -= tasa * d_n1

        # Ajustar neurona 2
        d_n2 = d_error_pred * salida["p2"] * relu_derivada(z2)
        n2["pt"] -= tasa * d_n2 * t
        n2["ph"] -= tasa * d_n2 * h
        n2["b"]  -= tasa * d_n2

        # Ajustar neurona 3
        d_n3 = d_error_pred * salida["p3"] * relu_derivada(z3)
        n3["pt"] -= tasa * d_n3 * t
        n3["ph"] -= tasa * d_n3 * h
        n3["b"]  -= tasa * d_n3

    if epoca % 500 == 0:
        print(f"Época {epoca} | Error total: {error_total:.2f}")

print("\nEntrenamiento terminado.\n")
print("Predicciones finales:")
for t, h, real in zip(tamanos_orig, habitaciones_orig, precios_orig):
    tn, hn = t / 100, h / 10
    z1 = tn * n1["pt"] + hn * n1["ph"] + n1["b"]
    z2 = tn * n2["pt"] + hn * n2["ph"] + n2["b"]
    z3 = tn * n3["pt"] + hn * n3["ph"] + n3["b"]
    s1, s2, s3 = relu(z1), relu(z2), relu(z3)
    pred = (s1 * salida["p1"] + s2 * salida["p2"] + s3 * salida["p3"] + salida["b"]) * 100
    print(f"Tamaño: {t}m2, {h} hab -> Predicción: {pred:.1f} | Real: {real}")