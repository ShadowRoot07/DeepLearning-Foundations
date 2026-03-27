import torch
import torch.nn as nn
import torch.optim as optim

# 1. Definimos una Red Neuronal como una Clase (Estándar Pro)
class MiPrimeraRed(nn.Module):
    def __init__(self):
        super(MiPrimeraRed, self).__init__()
        # Capa 1: Recibe 1 dato y lo pasa a 2 neuronas "ocultas"
        self.oculta = nn.Linear(1, 2)
        # Activación: Introduce "no linealidad"
        self.relu = nn.ReLU()
        # Capa 2: Toma las 2 salidas de la anterior y da 1 resultado
        self.salida = nn.Linear(2, 1)

    def forward(self, x):
        x = self.oculta(x)
        x = self.relu(x)
        x = self.salida(x)
        return x

# 2. Datos y Preparación
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

modelo = MiPrimeraRed()
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.1) # Adam es un mecánico más inteligente que SGD

# 3. Entrenamiento Intenso (500 épocas para que las 2 neuronas se coordinen)
print("--- Entrenando Red de 2 Neuronas ---")
for epoch in range(500):
    y_pred = modelo(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Época {epoch+1}: Pérdida = {loss.item():.4f}")

# 4. Prueba
prediccion = modelo(torch.tensor([[5.0]]))
print(f"\nPredicción para 5 horas: {prediccion.item():.4f}")

