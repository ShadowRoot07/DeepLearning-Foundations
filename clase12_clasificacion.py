import torch
import torch.nn as nn
import torch.optim as optim

# 1. Datos: Horas de estudio vs Resultado (0=Reprueba, 1=Aprueba)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)

class RedClasificadora(nn.Module):
    def __init__(self):
        super(RedClasificadora, self).__init__()
        self.oculta = nn.Linear(1, 4) # 4 neuronas para más capacidad
        self.sigmoid_hidden = nn.Sigmoid() # Usamos Sigmoid para evitar que mueran
        self.salida = nn.Linear(4, 1)
        self.sigmoid_final = nn.Sigmoid() # Salida final entre 0 y 1

    def forward(self, x):
        x = self.sigmoid_hidden(self.oculta(x))
        x = self.sigmoid_final(self.salida(x))
        return x

modelo = RedClasificadora()
# Para clasificación se usa BCELoss (Binary Cross Entropy)
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.1)

print("--- Entrenando Clasificador ---")
for epoch in range(200):
    y_pred = modelo(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Época {epoch+1}: Pérdida = {loss.item():.4f}")

# 3. Pruebas de fuego
with torch.no_grad():
    test_1 = modelo(torch.tensor([[1.5]])) # Debería ser cerca de 0
    test_2 = modelo(torch.tensor([[3.5]])) # Debería ser cerca de 1

print(f"\nProbabilidad con 1.5 horas: {test_1.item():.4f} (Reprobado)")
print(f"Probabilidad con 3.5 horas: {test_2.item():.4f} (Aprobado)")

