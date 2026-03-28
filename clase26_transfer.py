import torch
import torch.nn as nn

# 1. Imaginemos que descargamos un modelo "BERT-Base" ya entrenado
# Este modelo devuelve un vector de 768 números por cada frase.
class ModeloPreentrenado(nn.Module):
    def __init__(self):
        super(ModeloPreentrenado, self).__init__()
        # Esto simula las 12 capas de un Transformer gigante
        self.backbone = nn.Sequential(
            nn.Linear(10, 768), 
            nn.ReLU(),
            nn.Linear(768, 768)
        )

    def forward(self, x):
        return self.backbone(x)

# 2. NUESTRA PARTE: El "Cabezal" de clasificación
class ShadowClassifier(nn.Module):
    def __init__(self, base_model):
        super(ShadowClassifier, self).__init__()
        self.base = base_model
        # CONGELAMOS el modelo base para no arruinar lo que ya aprendió
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Solo entrenamos esta última capa pequeña
        self.fc = nn.Linear(768, 2) # 2 clases: ¿Comando seguro o peligroso?

    def forward(self, x):
        features = self.base(x)
        return self.fc(features)

# 3. Prueba de flujo
base = ModeloPreentrenado()
mi_ia = ShadowClassifier(base)

entrada = torch.randn(1, 10) # Simulación de una frase procesada
salida = mi_ia(entrada)

print(f"Modelo cargado y listo.")
print(f"¿Requiere gradiente el modelo base?: {next(base.parameters()).requires_grad}")
print(f"¿Requiere gradiente la capa final?: {mi_ia.fc.weight.requires_grad}")
print(f"\nSalida (Logits de seguridad): {salida}")

