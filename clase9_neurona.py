import torch
import torch.nn as nn # nn es Neural Networks

# 1. Definimos la entrada (Input)
# Imagina que X es el número de horas que estudias
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# 2. Definimos la salida esperada (Target)
# Queremos que la IA aprenda que el resultado es el doble (Y = 2 * X)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# 3. Creamos la neurona (Capa Lineal)
# 1 entrada (X) y 1 salida (y)
modelo = nn.Linear(in_features=1, out_features=1)

print(f"Pesos iniciales (aleatorios): {modelo.weight.item():.4f}")
print(f"Sesgo inicial (aleatorio): {modelo.bias.item():.4f}")

# 4. Hacemos una predicción ANTES de entrenar
prediccion_inicial = modelo(torch.tensor([[5.0]]))
print(f"\nSi estudio 5 horas, la IA sin entrenar predice: {prediccion_inicial.item():.4f}")

