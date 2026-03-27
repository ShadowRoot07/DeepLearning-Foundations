import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Definir cómo queremos transformar las imágenes al entrar
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Centrar los datos para que aprendan mejor
])

# 2. Descargar el Dataset de números escritos a mano
# train=True significa que son los datos para que la IA estudie
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 3. El DataLoader: Mezcla los datos y los entrega en grupos de 64
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 4. Probar el asistente
imágenes, etiquetas = next(iter(train_loader))

print(f"Cargamos un 'Batch' de: {imágenes.shape}") # [64, 1, 28, 28]
print(f"Etiquetas de este grupo: {etiquetas[:10]}") # Los primeros 10 números

