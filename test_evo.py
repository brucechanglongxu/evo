from evo import Evo
import torch

device = 'cuda:0'  # Adjust if using a different GPU or 'cpu' for CPU

# Load the model
evo_model = Evo('evo-1-131k-base')  # Genome-scale model
model, tokenizer = evo_model.model, evo_model.tokenizer
model.to(device)
model.eval()

# Input DNA sequence
sequence = 'ACGT'
input_ids = torch.tensor(
    tokenizer.tokenize(sequence),
    dtype=torch.int,
).to(device).unsqueeze(0)

# Perform inference
with torch.no_grad():
    logits, _ = model(input_ids)  # Outputs logits
print('Logits:', logits)
print('Shape (batch, length, vocab):', logits.shape)

