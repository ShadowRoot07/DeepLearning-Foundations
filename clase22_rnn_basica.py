import torch
import torch.nn as nn

# 1. Configuración: 10 palabras posibles, cada una de 5 números
# La RNN tendrá una "memoria" (hidden state) de 3 números.
vocab_size = 10
embedding_dim = 5
hidden_dim = 3

class MiniRNN(nn.Module):
    def __init__(self):
        super(MiniRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=True para que el primer número sea el tamaño del lote
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [Batch, Longitud de frase]
        x = self.embedding(x) # [Batch, Seq, Embed]
        
        # output: contiene la memoria en cada paso
        # hidden: contiene la memoria FINAL (el resumen de toda la frase)
        output, hidden = self.rnn(x)
        
        # Usamos solo la memoria final para decidir
        resultado = self.fc(hidden.squeeze(0))
        return resultado

# 2. Prueba con una "frase" de 4 palabras
frase_falsa = torch.tensor([[1, 3, 0, 7]], dtype=torch.long)
modelo = MiniRNN()

pred, memoria_final = modelo.rnn(modelo.embedding(frase_falsa))

print(f"Entrada (IDs): {frase_falsa.shape}")
print(f"Memoria generada en cada paso:\n{pred}")
print(f"\nEstado final de la memoria (Resumen de la frase):\n{memoria_final}")

