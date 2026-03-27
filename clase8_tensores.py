import torch
import numpy as np

print(f"Versión de PyTorch: {torch.__version__}")

# 1. Crear un tensor desde una lista (Vector)
data = [1.0, 2.0, 3.0]
tensor_simple = torch.tensor(data)
print(f"\nTensor Simple: {tensor_simple} | Forma: {tensor_simple.shape}")

# 2. Crear una matriz de ceros y otra de números aleatorios
# Imagina esto como inicializar un buffer vacío en NeoVim
matriz_ceros = torch.zeros((3, 2)) # 3 filas, 2 columnas
matriz_random = torch.rand((3, 3)) # Pesos iniciales de una neurona

print(f"\nMatriz de Ceros:\n{matriz_ceros}")
print(f"\nMatriz Aleatoria (Pesos):\n{matriz_random}")

# 3. Operaciones Matemáticas (El "pensamiento" de la IA)
# La IA aprende sumando y multiplicando estas matrices
a = torch.tensor([10, 20])
b = torch.tensor([5, 2])
resultado = a * b # Multiplicación elemento a elemento
print(f"\nResultado de Operación: {resultado}")

# 4. El concepto clave: DEVICE
# PyTorch permite mover datos entre CPU y GPU (en Actions usaremos CPU)
print(f"\n¿Estamos usando GPU?: {torch.cuda.is_available()}")

