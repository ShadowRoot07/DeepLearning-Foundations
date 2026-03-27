import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# 1. Descargar una imagen de prueba (Puedes cambiar esta URL)
url = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# 2. Definir las "Transforms" (El traductor de imagen a números)
transformador = transforms.Compose([
    transforms.Resize((128, 128)), # Redimensionar para que no pese tanto
    transforms.ToTensor(),         # ¡Aquí ocurre la magia! Convierte a Tensor [0, 1]
])

tensor_img = transformador(img)

print(f"Dimensiones del Tensor [C, H, W]: {tensor_img.shape}")
print(f"Valor máximo (Brillo): {tensor_img.max()}")
print(f"Valor mínimo (Oscuridad): {tensor_img.min()}")

# 3. ¿Qué ve la IA en un píxel específico?
pixel_central = tensor_img[:, 64, 64] # Canal, Fila, Columna
print(f"\nColor en el centro (RGB): {pixel_central}")

