import torch
import torch.nn.functional as F
import tiktoken

# Parameters
num_return_sequences = 5
max_length = 30

# Load model and tokenizer
model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Sampling loop
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]  # Get logits for the last token
        print("Logits Shape:", logits.shape)
        print("Logits Min/Max:", logits.min(), logits.max())

        # Validate logits
        assert not torch.isnan(logits).any(), "NaN detected in logits!"
        assert not torch.isinf(logits).any(), "Infinity detected in logits!"

        # Apply temperature scaling
        temperature = 1.0
        logits = logits / temperature

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        print("Probs Shape:", probs.shape)
        print("Probs Min/Max:", probs.min(), probs.max())
        print("Sum of Probs:", probs.sum(dim=-1))

        # Validate probabilities
        assert not torch.isnan(probs).any(), "NaN detected in probabilities!"
        assert not torch.isinf(probs).any(), "Infinity detected in probabilities!"
        assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0)), "Probabilities do not sum to 1!"

        # Select top-k probabilities
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        print("Top-k Probabilities:", topk_probs)
        print("Sum of Top-k Probabilities:", topk_probs.sum(dim=-1))
        print("Min Top-k Probability:", topk_probs.min())
        print("Max Top-k Probability:", topk_probs.max())

        # Validate top-k probabilities
        assert not torch.isnan(topk_probs).any(), "NaN detected in top-k probabilities!"
        assert not torch.isinf(topk_probs).any(), "Infinity detected in top-k probabilities!"
        assert (topk_probs > 0).all(), "Zero or negative values detected in top-k probabilities!"

        # Clamp and normalize top-k probabilities
        topk_probs = torch.clamp(topk_probs, min=1e-8)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Validate top-k indices
        vocab_size = 50257  # GPT-2 vocabulary size
        assert (topk_indices >= 0).all() and (topk_indices < vocab_size).all(), "Invalid indices in topk_indices!"

        # Sample an index
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# Decode and print results
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)