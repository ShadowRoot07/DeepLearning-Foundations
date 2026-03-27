import torch
import torch.nn as nn
import torch.nn.functional as F

class MiPrimeraCNN(nn.Module):
    def __init__(self):
        super(MiPrimeraCNN, self).__init__()
        # 1. Capa Convolucional: recibe 3 canales (RGB), saca 16 filtros
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 2. Max Pooling: Reduce la imagen a la mitad (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3. Segunda Convolución: de 16 filtros pasamos a 32
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 4. Capa Totalmente Conectada (Linear) para decidir qué es
        self.fc1 = nn.Linear(32 * 62 * 62, 10) # 10 categorías posibles

    def forward(self, x):
        # Aplicamos Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Aplicamos Conv2 -> ReLU -> Pool (aunque aquí solo haremos conv por brevedad)
        x = F.relu(self.conv2(x))
        # Aplanamos los datos para la capa final (de cubo a línea)
        x = x.view(-1, 32 * 62 * 62)
        x = self.fc1(x)
        return x

# Prueba con una "Imagen Fantasma" (puro ruido)
modelo = MiPrimeraCNN()
imagen_fake = torch.rand(1, 3, 128, 128) # Batch, RGB, 128x128
resultado = modelo(imagen_fake)

print(f"Forma de la imagen de entrada: {imagen_fake.shape}")
print(f"Salida de la CNN (10 probabilidades): {resultado.shape}")
print(f"\nPredicción cruda (Logits):\n{resultado}")

