import torch
import torch.nn as nn
import torch.nn.functional as F

# Frase de ejemplo
palabras = ['hola', 'como', 'estas', 'hoy']
vocabulario = {p: i for i, p in enumerate(palabras)}
ids = torch.tensor([vocabulario[p] for p in palabras])

# Embeddings: cada palabra representada con 5 números
dim_embedding = 5
capa_embedding = nn.Embedding(num_embeddings=len(vocabulario), embedding_dim=dim_embedding)
embeddings = capa_embedding(ids)  # forma: [4 palabras, 5 números]

print("Embeddings originales:")
for p, v in zip(palabras, embeddings):
    print(f"{p}: {v.tolist()}")
# Capas que transforman los embeddings en Q, K, V
capa_query = nn.Linear(dim_embedding, dim_embedding)
capa_key = nn.Linear(dim_embedding, dim_embedding)
capa_value = nn.Linear(dim_embedding, dim_embedding)

Q = capa_query(embeddings)
K = capa_key(embeddings)
V = capa_value(embeddings)

print("\nQuery (Q):")
print(Q)
print("\nKey (K):")
print(K)
# Comparamos Q contra K (producto punto) para saber qué tanto se "relacionan"
scores = torch.matmul(Q, K.T)  # cada palabra comparada contra todas

# Escalamos (truco estándar en Transformers para estabilidad numérica)
scores = scores / (dim_embedding ** 0.5)

# Convertimos en "porcentajes de atención" con softmax
pesos_atencion = F.softmax(scores, dim=-1)

print("\nPesos de atención (cada fila = cuánto atiende esa palabra a las demás):")
for palabra, fila in zip(palabras, pesos_atencion):
    valores = [f"{p}: {v:.2f}" for p, v in zip(palabras, fila)]
    print(f"{palabra} -> {valores}")
#Usamos los peso de atención para ponderar los valores (V)
salida_atencion = torch.matmul(pesos_atencion, V)

print("\nRepresentacion final de cada palabra(ya con contexto incorporado):")
for palabra, v in zip(palabras, salida_atencion):
    print(f"{palabra}: {v.tolist()}")
