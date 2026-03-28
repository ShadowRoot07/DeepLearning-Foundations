import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Preparación de Datos
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 2. Arquitectura de la CNN (Simplificada para velocidad)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320) # Aplanar
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

modelo = Net()
optimizer = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

# 3. Bucle de Entrenamiento (Solo 1 época para probar)
print("--- Iniciando Entrenamiento de la CNN ---")
modelo.train()
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = modelo(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 200 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)}: Pérdida = {loss.item():.4f}")
    if batch_idx > 400: break # Detenemos temprano para no saturar el log

# 4. Verificación
modelo.eval()
test_img, test_label = next(iter(train_loader))
with torch.no_grad():
    output = modelo(test_img[0].unsqueeze(0))
    prediccion = output.data.max(1, keepdim=True)[1]

print(f"\n--- Prueba Final ---")
print(f"Número Real: {test_label[0].item()}")
print(f"Predicción de la IA: {prediccion.item()}")

