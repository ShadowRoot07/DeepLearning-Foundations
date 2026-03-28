import torch
import torch.nn as nn

# 1. Nuestro modelo base
class ShadowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(x)

modelo = ShadowNet()
modelo.eval() # Siempre en eval() para optimizar

# 2. CUANTIZACIÓN DINÁMICA
# Convierte los pesos de 32 bits a 8 bits. ¡Reducción de tamaño 4x!
modelo_lite = torch.quantization.quantize_dynamic(
    modelo, {nn.Linear}, dtype=torch.qint8
)

# 3. TORCHSCRIPT (JIT - Just In Time)
# Crea un grafo optimizado del modelo
input_dummy = torch.randn(1, 512)
modelo_script = torch.jit.trace(modelo_lite, input_dummy)

# 4. GUARDAR EL MODELO OPTIMIZADO
modelo_script.save("shadow_model_mobile.pt")

print(f"--- Optimización de ShadowRoot07 ---")
print(f"Modelo original (Linear): {modelo.fc.weight.dtype}")
print(f"Modelo optimizado guardado como: shadow_model_mobile.pt")
print(f"¡Listo para ser invocado desde cualquier script de Termux!")

