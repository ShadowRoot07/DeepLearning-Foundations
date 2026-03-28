import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dataset: Secuencia de comandos que sueles usar
# "git add", "git commit", "git push"
vocab = {"<PAD>": 0, "git": 1, "add": 2, "commit": 3, "push": 4, "nvim": 5, "python": 6}
idx_to_vocab = {v: k for k, v in vocab.items()}

# Entrada: [git, add, commit] -> Salida esperada: [add, commit, push]
X = torch.tensor([[1, 2, 3]], dtype=torch.long)
Y = torch.tensor([[2, 3, 4]], dtype=torch.long) 

# 2. Modelo GRU (Una RNN con mejor memoria)
class ShadowGenerator(nn.Module):
    def __init__(self):
        super(ShadowGenerator, self).__init__()
        self.embedding = nn.Embedding(len(vocab), 8)
        self.gru = nn.GRU(8, 16, batch_first=True)
        self.fc = nn.Linear(16, len(vocab))

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

modelo = ShadowGenerator()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.01)

# 3. Entrenamiento
for epoch in range(200):
    hidden = None
    output, hidden = modelo(X, hidden)
    # Re-formatear para la pérdida: [Batch*Seq, Vocab]
    loss = criterion(output.view(-1, len(vocab)), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Generación: Le damos "git" y que complete
input_test = torch.tensor([[1]], dtype=torch.long) # "git"
hidden = None
print("--- Shadow Generator ---")
print("Input: git", end=" ")

with torch.no_grad():
    for _ in range(3):
        out, hidden = modelo(input_test, hidden)
        next_id = out.argmax(dim=2)[:, -1]
        print(idx_to_vocab[next_id.item()], end=" ")
        input_test = next_id.unsqueeze(0)
print("\n")

