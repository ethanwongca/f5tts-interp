# ==============================================================================
# f5_tts/model/mmdit.py (MODIFIED VERSION WITH HOOKS)
# ==============================================================================

# Diagnostic print to confirm this modified file is being loaded by Python
print("\n\n<<<<<<<<<< LOADING MODIFIED MMDiT BACKBONE WITH HOOKS >>>>>>>>>>\n\n")


from __future__ import annotations

import math
import torch
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    MMDiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# =======================================
# HOOK CLASSES (External Logic)
# =======================================

class FeatureExtractionHook:
    """
    Creates a PyTorch forward hook to capture a module's output tensor.
    """
    def __init__(self, storage: dict, name: str, output_index: int = 0):
        self.storage = storage
        self.name = name
        self.output_index = output_index

    def __call__(self, module, input, output):
        is_tuple = isinstance(output, tuple)
        target_tensor = output[self.output_index] if is_tuple else output

        self.storage[self.name] = {
            'tensor': target_tensor.clone().detach().cpu(),
            'device': target_tensor.device,
            'shape': target_tensor.shape
        }
        # print(f"--- Feature Extracted: '{self.name}' from module '{module.__class__.__name__}' ---")


class FeaturePatchHook:
    """
    Creates a PyTorch forward hook to modify (patch) a module's output tensor.
    """
    def __init__(self, batch_index: int | None = None, feature_index: int | None = None, output_index: int = 0, patch_vec: torch.Tensor | float = 0.0):
        self.batch_index = batch_index
        self.feature_index = feature_index
        self.output_index = output_index
        self.patch_vec = patch_vec

    def __call__(self, module, input, output):
        is_tuple = isinstance(output, tuple)
        target_tensor = output[self.output_index] if is_tuple else output

        # Ensure tensor is modifiable
        patched_tensor = target_tensor.clone()

        # Apply patch based on tensor dimensions
        if patched_tensor.ndim == 3: # (batch, seq, feature)
            self._patch_3d(patched_tensor)
        elif patched_tensor.ndim == 2: # (batch, feature)
            self._patch_2d(patched_tensor)
        else:
            print(f"Warning: Hook not applied. Unhandled tensor dimension: {patched_tensor.ndim}")
            return output

        # Return the modified output in its original format (tuple or tensor)
        if is_tuple:
            output_list = list(output)
            output_list[self.output_index] = patched_tensor
            return tuple(output_list)
        else:
            return patched_tensor

    def _patch_3d(self, tensor: torch.Tensor):
        if self.batch_index is not None and self.feature_index is not None:
            tensor[self.batch_index, :, self.feature_index] = self.patch_vec
        elif self.batch_index is not None:
            tensor[self.batch_index, :, :] = self.patch_vec
        elif self.feature_index is not None:
            tensor[:, :, self.feature_index] = self.patch_vec
        else:
            tensor[:, :, :] = self.patch_vec

    def _patch_2d(self, tensor: torch.Tensor):
        if self.batch_index is not None and self.feature_index is not None:
            tensor[self.batch_index, self.feature_index] = self.patch_vec
        elif self.batch_index is not None:
            tensor[self.batch_index, :] = self.patch_vec
        elif self.feature_index is not None:
            tensor[:, self.feature_index] = self.patch_vec
        else:
            tensor[:, :] = self.patch_vec

# =======================================
# EMBEDDING LAYERS (Unchanged)
# =======================================
class TextEmbedding(nn.Module):
    def __init__(self, out_dim, text_num_embeds, mask_padding=True):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)
        self.mask_padding = mask_padding
        self.precompute_max_pos = 1024
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: torch.Tensor, drop_text=False) -> torch.Tensor:
        text = text + 1
        text_mask = text == 0 if self.mask_padding else None
        if drop_text:
            text = torch.zeros_like(text)
        text = self.text_embed(text)
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, text.shape[1], max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]
        text = text + text_pos_embed
        if self.mask_padding:
            text = text.masked_fill(text_mask.unsqueeze(-1), 0.0)
        return text

class AudioEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x

# =======================================
# MMDiT BACKBONE WITH HOOK FUNCTIONALITY
# =======================================

class MMDiT(nn.Module):
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

    # --- Hook Management Methods ---

    def _get_submodule(self, path: str) -> nn.Module:
        """Helper to retrieve a submodule from a path string (e.g., 'transformer_blocks.0.attn')."""
        module = self
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def add_extraction_hook(self, submodule_path: str, hook_name: str, output_index: int = 0):
        """Attaches a feature extraction hook to a submodule."""
        if hook_name in self.hook_handles:
            self.hook_handles[hook_name].remove()
        try:
            submodule = self._get_submodule(submodule_path)
            handle = submodule.register_forward_hook(
                FeatureExtractionHook(self.extracted_activations, hook_name, output_index)
            )
            self.hook_handles[hook_name] = handle
            print(f"✅ Added extraction hook '{hook_name}' to '{submodule_path}'.")
        except Exception as e:
            print(f"❌ ERROR adding extraction hook '{hook_name}' to '{submodule_path}': {e}")

    def add_patch_hook(self, submodule_path: str, hook_name: str, **patch_kwargs):
        """Attaches a feature patching hook to a submodule."""
        if hook_name in self.hook_handles:
            self.hook_handles[hook_name].remove()
        try:
            submodule = self._get_submodule(submodule_path)
            handle = submodule.register_forward_hook(FeaturePatchHook(**patch_kwargs))
            self.hook_handles[hook_name] = handle
            print(f"✅ Added patch hook '{hook_name}' to '{submodule_path}'.")
        except Exception as e:
            print(f"❌ ERROR adding patch hook '{hook_name}' to '{submodule_path}': {e}")

    def remove_hooks(self):
        """Removes all attached hooks."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        print("Hooks removed.")

    def clear_activations(self):
        """Clears the dictionary of extracted activations."""
        self.extracted_activations.clear()

    def __str__(self):
        """Prints a summary of the captured activations."""
        report = "--- Captured Activations Summary ---\n"
        if not self.extracted_activations:
            return report + "  (No activations captured)\n"
        for name, data in self.extracted_activations.items():
            tensor = data['tensor']
            report += (
                f"  - '{name}':\t"
                f"Shape={data['shape']}, "
                f"Mean={tensor.mean():.4f}, "
                f"Std={tensor.std():.4f}\n"
            )
        return report

    # --- Main Forward Pass ---

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ):
        self.clear_activations() # Clear previous results at the start of each pass
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
            if mask is not None:
                mask = torch.cat((mask, mask), dim=0)
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