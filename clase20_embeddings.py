import torch
import torch.nn as nn

# 1. Nuestro vocabulario (El diccionario del Grimorio)
# Cada palabra tiene un índice (ID)
vocabulario = {
    "shadow": 0,
    "root": 1,
    "programacion": 2,
    "ia": 3,
    "termux": 4,
    "error": 5
}

# 2. Definimos la capa de Embedding
# Queremos que cada una de las 6 palabras sea un vector de 3 números
# (En modelos reales son vectores de 512, 768 o más)
embedding_layer = nn.Embedding(num_embeddings=6, embedding_dim=3)

# 3. Convertimos una frase en "IDs"
# Frase: "shadow programacion termux"
frase_indices = torch.tensor([vocabulario["shadow"], 
                             vocabulario["programacion"], 
                             vocabulario["termux"]])

# 4. ¡La magia! Pasamos los IDs por la capa de Embedding
vectores_de_palabras = embedding_layer(frase_indices)

print(f"Frase original (IDs): {frase_indices}")
print(f"\nFrase convertida en vectores (Significado numérico):\n{vectores_de_palabras}")

# 5. ¿Qué tan parecidas son las palabras? (Producto punto simple)
similitud = torch.dot(vectores_de_palabras[0], vectores_de_palabras[1])
print(f"\nSimilitud entre 'shadow' y 'programacion': {similitud.item():.4f}")

