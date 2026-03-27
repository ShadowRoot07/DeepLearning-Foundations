import torch
import torch.nn as nn
import torch.optim as optim

# 1. Datos (X = Horas, y = Nota esperada)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# 2. Modelo, Pérdida y Optimizador
modelo = nn.Linear(1, 1)
criterion = nn.MSELoss() 
# lr = Learning Rate (qué tan grandes son los pasos del mecánico)
optimizer = optim.SGD(modelo.parameters(), lr=0.01) 

print("--- Iniciando Entrenamiento ---")

# 3. El Bucle Mágico
for epoch in range(100):
    # Forward pass: Predecir
    y_pred = modelo(X)
    
    # Calcular el error
    loss = criterion(y_pred, y)
    
    # Backpropagation: El mecánico analiza el error
    optimizer.zero_grad() # Limpiar residuos anteriores
    loss.backward()       # Calcular dirección del ajuste
    optimizer.step()      # Ajustar peso y sesgo
    
    if (epoch+1) % 20 == 0:
        print(f"Época {epoch+1}: Pérdida = {loss.item():.4f}")

# 4. Prueba Final
print("\n--- Entrenamiento Finalizado ---")
peso_final = modelo.weight.item()
prediccion = modelo(torch.tensor([[5.0]]))

print(f"Nuevo peso: {peso_final:.4f} (Objetivo: 2.0)")
print(f"Si estudio 5 horas, ahora la IA predice: {prediccion.item():.4f}")

