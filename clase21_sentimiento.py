import torch
import torch.nn as nn
import torch.optim as optim

# 1. Vocabulario y Datos de entrenamiento
vocab = {"feliz": 0, "amo": 1, "termux": 2, "error": 3, "odio": 4, "funciona": 5, "mal": 6}

# Frases: [feliz, amo, funcionan] (Positivas) | [odio, error, mal] (Negativas)
X = torch.tensor([[0, 1, 5], [4, 3, 6]], dtype=torch.long) 
y = torch.tensor([[1.0], [0.0]], dtype=torch.float32) # 1=Positivo, 0=Negativo

# 2. Arquitectura de la Red de Texto
class ShadowSentiment(nn.Module):
    def __init__(self):
        super(ShadowSentiment, self).__init__()
        # 7 palabras en el vocabulario, cada una vector de 4 dimensiones
        self.embedding = nn.Embedding(7, 4)
        self.fc = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1) # Promediar los vectores de las palabras de la frase
        x = self.fc(x)
        return self.sigmoid(x)

modelo = ShadowSentiment()
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.1)

# 3. Entrenamiento rápido
print("--- Entrenando Detector de Sentimiento ---")
for epoch in range(100):
    y_pred = modelo(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Prueba de Fuego: "¿Odio el error?"
# IDs: odio=4, el(no está, usamos termux=2), error=3 -> [4, 2, 3]
test_frase = torch.tensor([[4, 2, 3]], dtype=torch.long)
pred = modelo(test_frase)

print(f"\nFrase: 'odio termux error'")
print(f"Probabilidad de ser POSITIVA: {pred.item()*100:.2f}%")
print("Veredicto:", "Positivo" if pred.item() > 0.5 else "Negativo")

