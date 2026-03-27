import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# 1. Cargar la imagen (Usando tu link o uno de respaldo)
url = "https://i.pinimg.com/originals/05/65/d6/0565d6d6741757f587d60517866380c2.jpg" # Ejemplo de Pinterest
try:
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert('L') # Convertimos a Blanco y Negro ('L')
except:
    print("Error con la URL, usando imagen aleatoria...")
    img = Image.fromarray((torch.rand(256, 256) * 255).numpy().astype('uint8'))

transform = transforms.ToTensor()
tensor_img = transform(img).unsqueeze(0) # Añadimos una dimensión extra para el 'Batch'

# 2. Definir un Filtro de Sobel (Detecta bordes verticales)
# Es una matriz 3x3 que resalta cambios bruscos de color
sobel_kernel = torch.tensor([[[[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]]]], dtype=torch.float32)

# 3. Aplicar la Convolución
bordes = F.conv2d(tensor_img, sobel_kernel, padding=1)

print(f"Dimensiones originales: {tensor_img.shape}")
print(f"Dimensiones tras detectar bordes: {bordes.shape}")

# 4. Análisis de Intensidad
print(f"\nIntensidad máxima de bordes detectados: {bordes.max().item():.4f}")
if bordes.max() > 0.5:
    print("¡Bordes fuertes detectados! La IA encontró contrastes marcados.")
else:
    print("Bordes suaves. La imagen es muy plana o borrosa.")

