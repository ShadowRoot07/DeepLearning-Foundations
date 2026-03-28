import torch
import torch.nn as nn

# 1. Configuración del "Mini-GPT"
# d_model: tamaño del embedding (512 es el estándar, usamos 32 para pruebas)
# nhead: número de "cabezas" de atención (cuántos enfoques distintos tiene la IA)
embed_dim = 32
n_cabezas = 4 

class ShadowTransformer(nn.Module):
    def __init__(self):
        super(ShadowTransformer, self).__init__()
        # Capa de Atención: permite que las palabras "se miren" entre sí
        self.atencion = nn.MultiheadAttention(embed_dim, n_cabezas, batch_first=True)
        self.fc = nn.Linear(embed_dim, 10) # Salida para 10 palabras posibles

    def forward(self, x):
        # En el Transformer, x entra como embeddings
        # La atención devuelve (valores_atendidos, pesos_de_atencion)
        attn_output, attn_weights = self.atencion(x, x, x) # Query, Key, Value
        out = self.fc(attn_output[:, -1, :]) # Predicción basada en la última palabra
        return out, attn_weights

# 2. Prueba con una frase de 5 palabras
frase_fake = torch.rand(1, 5, embed_dim) # [Batch, Seq, Embed]
modelo = ShadowTransformer()
prediccion, pesos = modelo(frase_fake)

print(f"Dimensiones de los pesos de atención: {pesos.shape}")
print(f"Interpretación: Una matriz de {pesos.shape[1]}x{pesos.shape[2]} donde cada palabra mira a las demás.")

