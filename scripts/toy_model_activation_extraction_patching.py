'''
Patching and Activation Hook Toy Model

If the ReLu output is all 0's, this indicates patching works:
Patch linear layer (1) as all 0s, ReLu is max(x, 0) applied element-wise
'''

import torch
import torch.nn as nn

class PatchVectorTensor:
    def __init__(self, feature_number: int, patch_vector=None):
        """
        Args:
            feature_number: which feature (dimension) to patch
            patch_vector: replacement values (defaults to zeros)
        """
        self.feature_number = feature_number
        self.patch_vector = patch_vector

    def __call__(self, module, input, output):
        """
        Works for both plain tensors and LSTM outputs (tuple).
        """
        if isinstance(output, tuple):
            # Handle LSTM: (out, (h_n, c_n))
            out, (h_n, c_n) = output

            # patch 'out' along the last dimension
            n, s, m = out.size()
            if self.patch_vector is None:
                self.patch_vector = torch.zeros(n, s, device=out.device)
            out_patched = out.clone()
            out_patched[:, :, self.feature_number] = self.patch_vector

            print(f"[Patch Hook] Patched feature {self.feature_number} in LSTM output")
            return out_patched, (h_n, c_n)
        else:
            # Handle simple 2D tensor (like Linear output)
            n, m = output.size()
            if self.patch_vector is None:
                self.patch_vector = torch.zeros(n, device=output.device)
            patched = output.clone()
            patched[:, self.feature_number] = self.patch_vector
            print(f"[Patch Hook] Patched feature {self.feature_number} in {module.__class__.__name__}")
            return patched


# Toy model with LSTM added
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)
        self.rnn = nn.LSTM(input_size=5, hidden_size=8, batch_first=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        # Expand to 3D for LSTM
        x = x.unsqueeze(1)  # (batch, seq_len=1, features=5)
        out, (h_n, c_n) = self.rnn(x)
        return out, (h_n, c_n)


# Instantiate model
model = ToyModel()

# Example input
x = torch.randn(1, 10)

# Activation store
activations = {}

def hook_fn(module, input, output):
    if isinstance(output, tuple):
        activations[module.__class__.__name__] = [o.detach().cpu() for o in (output[0], *output[1])]
    else:
        activations[module.__class__.__name__] = output.detach().cpu()

# Register hooks
hook_lin = model.linear1.register_forward_hook(hook_fn)
patch_lin = model.linear1.register_forward_hook(PatchVectorTensor(0))  # patch feature 0 of linear1
hook_relu = model.relu.register_forward_hook(hook_fn)
hook_rnn = model.rnn.register_forward_hook(hook_fn)
patch_rnn = model.rnn.register_forward_hook(PatchVectorTensor(5))      # patch feature 1 of LSTM output

# Forward pass
out, (h_n, c_n) = model(x)

# Print results
print("\n=== Final Model Output ===")
print("out:", out)
print("h_n:", h_n)
print("c_n:", c_n)

print("\n=== Captured Activations ===")
for k, v in activations.items():
    if isinstance(v, list):
        print([t for t in v])
    else:
        print(v)

# Cleanup
hook_lin.remove()
patch_lin.remove()
hook_relu.remove()
hook_rnn.remove()
patch_rnn.remove()
