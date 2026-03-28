import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Definimos la misma arquitectura que usamos para entrenar
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
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Cargar imagen local de Subaru
try:
    img = Image.open("subaru.jpg").convert('L') # Convertir a blanco y negro
except FileNotFoundError:
    print("Error: No encontré subaru.jpg. Verifica el nombre.")
    exit()

# 3. Transformaciones para que parezca un número MNIST
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

tensor_subaru = transform(img).unsqueeze(0) # [1, 1, 28, 28]

# 4. Inferencia
modelo = Net() # Nota: Aquí el modelo está vacío (pesos al azar)
modelo.eval()

with torch.no_grad():
    output = modelo(tensor_subaru)
    probabilidades = torch.nn.functional.softmax(output, dim=1)
    prediccion = output.data.max(1, keepdim=True)[1]

print(f"--- Diagnóstico de Subaru ---")
print(f"La IA cree que Subaru es el número: {prediccion.item()}")
print(f"Confianza en esa decisión: {probabilidades.max().item()*100:.2f}%")

