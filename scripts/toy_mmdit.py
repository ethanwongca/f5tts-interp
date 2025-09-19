"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

# Import future annotations for type hinting compatibility (Python 3.7+)
from __future__ import annotations

# Import standard libraries and PyTorch modules
import math
import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

# Import various modules and layers from the F5-TTS codebase
from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    MMDiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# ---------------------------------------
# HOOK CLASSES (External Logic)
# --------------------------------------

class FeatureExtractionHook:
    """
    Creates a PyTorch forward hook that can extract any particular feature from a module's output.
    """
    def __init__(self, storage: dict, name: str, output_index: int = 0):
        self.storage = storage
        self.name = name
        self.output_index = output_index

    def __call__(self, module, input, output):
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            if self.output_index >= len(output):
                print(f"Warning: Hook '{self.name}' failed. output_index is out of bounds.")
                return
            target_tensor = output[self.output_index]
        else:
            target_tensor = output

        self.storage[self.name] = {
            'tensor': target_tensor.clone().detach().cpu(),
            'device': target_tensor.device,
            'shape': target_tensor.shape
        }
        print(f"--- Feature Extracted: '{self.name}' from module '{module.__class__.__name__}' ---")


class FeaturePatchHook:
    """
    Creates a PyTorch forward hook that can patch any particular feature, for a particular batch.
    """
    def __init__(self, batch_index: int | None = None, feature_index: int | None = None, output_index: int = 0, patch_vec: torch.Tensor | float = 0.0):
        self.batch_index = batch_index
        self.feature_index = feature_index
        self.output_index = output_index
        self.patch_vec = patch_vec

    def __call__(self, module, input, output) -> torch.Tensor:
        is_tuple = isinstance(output, tuple)
        if is_tuple:
            if self.output_index >= len(output):
                raise IndexError(f"output_index {self.output_index} is out of bounds for tuple of size {len(output)}")
            target_tensor = output[self.output_index]
        else:
            target_tensor = output

        if target_tensor.ndim == 3:
            patched_tensor = self.forward_path_three_dimensions(target_tensor)
        elif target_tensor.ndim == 2:
            patched_tensor = self.forward_path_two_dimensions(target_tensor)
        else:
            print(f"Warning: Hook not applied. Unhandled dimension: {target_tensor.ndim}")
            return output

        if is_tuple:
            output_list = list(output)
            output_list[self.output_index] = patched_tensor
            return tuple(output_list)
        else:
            return patched_tensor

    def forward_path_three_dimensions(self, target_tensor: torch.Tensor) -> torch.Tensor:
        _batch_size, seq_len, _feature_size = target_tensor.shape
        if isinstance(self.patch_vec, float):
            patch_tensor = torch.full((seq_len,), self.patch_vec, device=target_tensor.device, dtype=target_tensor.dtype)
        else:
            patch_tensor = self.patch_vec.to(target_tensor.device, target_tensor.dtype)

        if patch_tensor.ndim != 1 or patch_tensor.shape[0] != seq_len:
            raise ValueError(f"For 3D tensors, patch_vec must be float or 1D tensor of length {seq_len}")

        if self.batch_index is None and self.feature_index is None:
            target_tensor[:, :, :] = patch_tensor.view(1, seq_len, 1)
        elif self.batch_index is None:
            target_tensor[:, :, self.feature_index] = patch_tensor
        elif self.feature_index is None:
            target_tensor[self.batch_index, :, :] = patch_tensor.view(seq_len, 1)
        else:
            target_tensor[self.batch_index, :, self.feature_index] = patch_tensor
        return target_tensor

    def forward_path_two_dimensions(self, target_tensor: torch.Tensor) -> torch.Tensor:
        if self.batch_index is None and self.feature_index is None:
            target_tensor[:, :] = self.patch_vec
        elif self.batch_index is None:
            target_tensor[:, self.feature_index] = self.patch_vec
        elif self.feature_index is None:
            target_tensor[self.batch_index, :] = self.patch_vec
        else:
            target_tensor[self.batch_index, self.feature_index] = self.patch_vec
        return target_tensor

# --- [TextEmbedding and AudioEmbedding classes would go here, unchanged] ---
class TextEmbedding(nn.Module):
    """
    Embeds text tokens into a continuous vector space, adds sinusoidal positional encoding,
    and optionally masks padding tokens.
    """
    def __init__(self, out_dim, text_num_embeds, mask_padding=True):
        super().__init__()
        # Embedding layer for text tokens (+1 for filler token at index 0)
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.mask_padding = mask_padding  # Whether to mask filler and batch padding tokens

        self.precompute_max_pos = 1024  # Maximum sequence length for precomputed positional embeddings
        # Register a buffer for precomputed sinusoidal frequencies (not a parameter, but saved with the model)
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"], drop_text=False) -> int["b nt d"]:  # noqa: F722
        """
        Args:
            text: Tensor of shape (batch, text_seq_len) with token indices. -1 is used for padding.
            drop_text: If True, zero out the text (for classifier-free guidance or ablation).
        Returns:
            Embedded text tensor of shape (batch, text_seq_len, out_dim)
        """
        text = text + 1  # Shift all tokens by 1 so 0 can be used as the filler token
        if self.mask_padding:
            text_mask = text == 0  # Mask for filler tokens (originally -1)

        if drop_text:  # If dropping text conditioning, zero out all tokens
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # Embed tokens: (b, nt) -> (b, nt, d)

        # Sinusoidal positional embedding
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)  # Start index for each batch
        batch_text_len = text.shape[1]  # Length of text sequence
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)  # (b, nt)
        text_pos_embed = self.freqs_cis[pos_idx]  # (b, nt, d)

        text = text + text_pos_embed  # Add positional embedding to token embedding

        if self.mask_padding:
            # Mask out filler tokens by setting their embeddings to zero
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return text
    
class AudioEmbedding(nn.Module):
    """
    Embeds audio features (input and conditioning audio) into a continuous space,
    applies a linear layer and convolutional positional embedding.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Linear layer to combine input and conditioning audio (concatenated)
        self.linear = nn.Linear(2 * in_dim, out_dim)
        # Convolutional positional embedding
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        """
        Args:
            x: Input audio features (batch, seq_len, in_dim)
            cond: Conditioning audio features (batch, seq_len, in_dim)
            drop_audio_cond: If True, zero out conditioning audio (for ablation)
        Returns:
            Embedded audio tensor (batch, seq_len, out_dim)
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)  # Remove conditioning audio if specified
        x = torch.cat((x, cond), dim=-1)  # Concatenate input and cond along feature dim
        x = self.linear(x)                # Project to out_dim
        x = self.conv_pos_embed(x) + x    # Add convolutional positional embedding
        return x

# ---------------------------------------------------
# REFACTORED MMDiT CLASS WITH INTEGRATED HOOKS
# ---------------------------------------------------

class MMDiT(nn.Module):
    """
    Multi-modal DiT backbone with integrated methods for feature extraction and patching.
    """
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_mask_padding=True,
        qk_norm=None,
    ):
        super().__init__()

        # --- Standard Model Layers ---
        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(dim, text_num_embeds, mask_padding=text_mask_padding)
        self.text_cond, self.text_uncond = None, None
        self.audio_embed = AudioEmbedding(mel_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.initialize_weights()

        # --- Integrated Hook Management ---
        self.hook_handles = {}
        self.extracted_activations = {}

    # --- [initialize_weights, get_input_embed, clear_cache methods are unchanged] ---
    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def get_input_embed(self, x, cond, text, drop_audio_cond: bool = False, drop_text: bool = False, cache: bool = True):
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)
        return x, c

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
        
    # --- UPDATED Hook Management Methods ---

    def _get_submodule(self, path: str) -> nn.Module:
        """Helper function to retrieve a submodule from a path string."""
        module = self
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def add_extraction_hook(self, submodule_path: str, hook_name: str, output_index: int = 0):
        """Attaches a feature extraction hook to a submodule using its path."""
        if hook_name in self.hook_handles:
            print(f"Warning: A hook with name '{hook_name}' already exists. Overwriting.")
            self.hook_handles[hook_name].remove()

        try:
            submodule = self._get_submodule(submodule_path)
        except (AttributeError, IndexError) as e:
            print(f"Error finding submodule at path '{submodule_path}': {e}")
            return

        hook = FeatureExtractionHook(storage=self.extracted_activations, name=hook_name, output_index=output_index)
        handle = submodule.register_forward_hook(hook)
        self.hook_handles[hook_name] = handle
        print(f"Added extraction hook '{hook_name}' to '{submodule_path}'.")

    def add_patch_hook(self, submodule_path: str, hook_name: str, **patch_kwargs):
        """Attaches a feature patching hook to a submodule using its path."""
        if hook_name in self.hook_handles:
            print(f"Warning: A hook with name '{hook_name}' already exists. Overwriting.")
            self.hook_handles[hook_name].remove()

        try:
            submodule = self._get_submodule(submodule_path)
        except (AttributeError, IndexError) as e:
            print(f"Error finding submodule at path '{submodule_path}': {e}")
            return

        hook = FeaturePatchHook(**patch_kwargs)
        handle = submodule.register_forward_hook(hook)
        self.hook_handles[hook_name] = handle
        print(f"Added patch hook '{hook_name}' to '{submodule_path}'.")

    def remove_hooks(self):
        """Removes all attached hooks."""
        for name, handle in self.hook_handles.items():
            handle.remove()
        self.hook_handles = {}
        print("Removed all hooks.")

    def clear_activations(self):
        """Clears the dictionary of extracted activations."""
        self.extracted_activations.clear()

    def __str__(self):
        """Prints a summary of the captured activations."""
        report = "--- Captured Activations ---\n"
        if not self.extracted_activations:
            report += "  (No activations captured yet)\n"
        else:
            for name, values in self.extracted_activations.items():
                tensor = values['tensor']
                report += (
                    f"  - Hook '{name}':\t"
                    f"Shape={values['shape']}, "
                    f"Device='{values['device']}', "
                    f"Mean={tensor.mean():.2f}\n"
                )
        report += "----------------------------"
        return report

    # --- [forward method is unchanged] ---
    def forward(self, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False, cfg_infer=False, cache=False):
        self.clear_activations()
        batch = x.shape[0]
        if time.ndim == 0:
            time = time.repeat(batch)
        t = self.time_embed(time)
        if cfg_infer:
            x_cond, c_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x, c = self.get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache)

        seq_len, text_len = x.shape[1], text.shape[1]
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)

        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text)

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output

