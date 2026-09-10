# Frase de ejemplo
frase = "hola como estas hoy"

# Partimos en palabras (tokenizar)
palabras = frase.split()
print("Palabras:", palabras)

# Creamos un "vocabulario": cada palabra única recibe un número (ID)
vocabulario = {}
for palabra in palabras:
    if palabra not in vocabulario:
        vocabulario[palabra] = len(vocabulario)

print("Vocabulario:", vocabulario)

# Convertimos la frase a números usando el vocabulario
ids = [vocabulario[p] for p in palabras]
print("Frase como números:", ids)

import torch
import torch.nn as nn

# Creamos una capa de embeddings:
# - 4 palabras en el vocabulario
# - cada palabra se representa con un vector de 5 números
capa_embedding = nn.Embedding(
    num_embeddings=len(vocabulario),
    embedding_dim=5
)

# Convertimos nuestra frase (ids) en tensor
tensor_ids = torch.tensor(ids)

# Obtenemos los embeddings de cada palabra
embeddings = capa_embedding(tensor_ids)

print("\nEmbeddings (cada fila es una palabra, representada con 5 números):")
for palabra, vector in zip(palabras, embeddings):
    print(f"{palabra}: {vector.tolist()}")
