import torch
import torch.nn as nn

# 1. Configuración de dimensiones
input_dim = 10  # Tamaño del vocabulario
embed_dim = 8   # Tamaño del vector de cada palabra
hidden_dim = 16 # Tamaño de la memoria interna
n_layers = 2    # ¡Ahora usamos 2 capas de LSTM una sobre otra!

class ShadowLSTM(nn.Module):
    def __init__(self):
        super(ShadowLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # La LSTM ahora es multicapa para entender patrones más complejos
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        # h0 y c0 son el estado oculto y la "celda" de memoria inicializados en cero
        out, (hn, cn) = self.lstm(x)
        # Tomamos el último paso de la secuencia
        out = self.fc(out[:, -1, :])
        return out

modelo = ShadowLSTM()
print(modelo)

# 2. Prueba de flujo
frase_test = torch.tensor([[1, 2, 5, 4]], dtype=torch.long) # IDs aleatorios
salida = modelo(frase_test)

print(f"\nEntrada: {frase_test.shape} (1 frase de 4 palabras)")
print(f"Salida: {salida.shape} (Predicción para las 10 palabras del vocabulario)")

