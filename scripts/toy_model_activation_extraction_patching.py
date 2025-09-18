'''
Patching and Activation Hook Toy Model

If the ReLu output is all 0's, this indicates patching works:
Patch linear layer (1) as all 0s, ReLu is max(x, 0) applied element-wise
'''

import torch
import torch.nn as nn

class PatchVectorTensor:
    def __init__(self, feature_number: int, patch_vector = None):
        """
        Initializes our patching token! 
        Args:
            feature_number: selects the particular feature in our tensor we want to patch 
            patch_vector: selects what we want to replace feature with
        """
        self.feature_number = feature_number 
        self.patch_vector = patch_vector 

    def __call__(self, module, input, output):
        """
        Calls our patch hook 
        Args:
            module: particular layer our hook is placed 
            input: input into this layer
            output: output to this layer 
        """
        # Assumes a 2-D Tensor 
        n, m = output.size() 

        if self.patch_vector is None:
            self.patch_vector = torch.zeros(n)

        output[:, self.feature_number] = self.patch_vector

        return output


# 1. Define a simple neural network class
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)  # Layer 1: Linear (input 10, output 20)
        self.relu = nn.ReLU()             # Activation: ReLU
        self.linear2 = nn.Linear(20, 5)   # Layer 2: Linear (input 20, output 5)

    def forward(self, x):
        x = self.linear1(x)  # Pass input through first linear layer
        x = self.relu(x)     # Apply ReLU activation
        x = self.linear2(x)  # Pass through second linear layer
        return x             # Output

# 2. Create an instance of the model
model = ToyModel()

# 3. Prepare a random input tensor (batch size 1, 10 features)
x = torch.randn(1, 10)

# 4. Set up a dictionary to store activations
activations = {}

# 5. Define a hook function to capture the output of linear1
#    The hook function takes three arguments: module, input, output
#    We store the output (activation) in the activations dictionary

def hook_fn(module, input, output):
    if isinstance(module, nn.Linear):
        activations['linear1'] = {
            'tensor': output.detach().cpu(),
            'device': output.device
        }
    elif isinstance (module, nn.ReLU):
        activations['relu'] = {
            'tensor': output.detach().cpu(),
            'device': output.device
        }

def patch_hook(module, input, output):
    '''
    Inputs:
        Module: The layer (i.e. Conv2D etc)
        Input: Inputs passed into the layer's forward method 
        Output: Output of the layer's method 
    '''
    print("\n[Patch Hook] Original linear1 output:", output)
    # Example: zero out the activation
    patched = torch.zeros_like(output)
    print("[Patch Hook] Patched linear1 output:", patched)
    return patched  # This will replace the output of linear1

patch_hook_new = PatchVectorTensor(0)
# 6. Register the hook to linear1
hook_handle = model.linear1.register_forward_hook(hook_fn)
patch_handle = model.linear1.register_forward_hook(patch_hook_new)
hook_handle2 = model.relu.register_forward_hook(hook_fn)

# 7. Run the model on the input
output = model(x)

# 8. Print the output of the model
print("Model output:", output)

# 9. Print the captured activations from linear1
print("Activations from linear1:", activations['linear1'])
print("Activations from linear1 tensor shape:", activations['linear1']['tensor'].shape)
print("Activations from relu:", activations['relu'])
print("Activations from relu tensor shape:", activations['relu']['tensor'].shape)

# 10. Remove the hook (good practice to avoid side effects)
hook_handle.remove() 
patch_handle.remove()
hook_handle2.remove()