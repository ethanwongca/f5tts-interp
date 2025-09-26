# ==============================================================================
# test_hooks_inference.py
# ==============================================================================

import argparse
import os
import pickle
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import soundfile as sf
import tomli
import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

# Make sure these utils are available in your environment
from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step,
    preprocess_ref_audio_text,
    speed,
    sway_sampling_coef,
    target_rms,
)

# === HOOK SETUP FUNCTION ===
def setup_model_hooks(model):
    """Adds extraction and patch hooks to the model for analysis."""
    print("\n" + "="*60)
    print("SETTING UP MODEL HOOKS FOR TESTING")
    print("="*60)
    
    if not hasattr(model, 'add_extraction_hook'):
        print("❌ ERROR: Model does not have hook functionality!")
        print("Make sure you're using the modified MMDiT backbone and that Python is loading it.")
        return False
    
    model.remove_hooks()
    model.clear_activations()
    
    # --- 1. EXTRACTION HOOKS (Capture intermediate tensors) ---
    print("Adding extraction hooks...")
    
    model.add_extraction_hook("text_embed", "text_embeddings")
    model.add_extraction_hook("audio_embed", "audio_embeddings")
    
    # IMPORTANT: Verify 'attn' and 'ff' names from the architecture printout.
    # They might be different, e.g., 'attention' or 'feed_forward'.
    model.add_extraction_hook("transformer_blocks.0.attn", "block0_attention_output", output_index=0)
    model.add_extraction_hook("transformer_blocks.0.attn", "block0_context_output", output_index=1)
    model.add_extraction_hook("transformer_blocks.0.ff", "block0_feedforward_output", output_index=0)
    
    if len(model.transformer_blocks) > 4:
        model.add_extraction_hook("transformer_blocks.4.attn", "block4_attention_output", output_index=0)
    
    model.add_extraction_hook("norm_out", "final_norm_output")
    model.add_extraction_hook("proj_out", "final_projection_output")
    
    # --- 2. PATCH HOOKS (Modify tensors during the forward pass) ---
    print("\nAdding patch hooks...")
    
    # Example: Zero out the first feature of the context attention in block 0
    model.add_patch_hook(
        submodule_path='transformer_blocks.0.attn', # VERIFY 'attn'
        hook_name='zero_context_patch',
        output_index=1,      # Target the second output (context attention)
        feature_index=0,     # Target the first feature dimension
        batch_index=0,       # Target the first item in the batch
        patch_vec=0.0        # Patch value
    )
    
    print("\n✓ Hook setup complete!")
    return True

def save_hook_results(model, output_dir):
    """Saves captured activations and generates an analysis report."""
    print("\n" + "="*60)
    print("ANALYZING AND SAVING HOOK RESULTS")
    print("="*60)
    
    # Print summary from the model's __str__ method
    print(str(model))
    
    # Save raw activations to a pickle file for deep analysis
    activation_file = Path(output_dir) / "hook_activations.pkl"
    with open(activation_file, 'wb') as f:
        pickle.dump(model.extracted_activations, f)
    print(f"\n✅ Raw activations saved to: {activation_file}")
    
    # Create a human-readable analysis report
    report_file = Path(output_dir) / "hook_analysis.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("F5-TTS Hook Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        for name, data in model.extracted_activations.items():
            tensor = data['tensor']
            f.write(f"HOOK: {name}\n")
            f.write(f"  - Shape: {data['shape']}\n")
            f.write(f"  - Device: {data['device']}\n")
            f.write(f"  - Stats (Mean/Std/Min/Max): {tensor.mean():.6f} / {tensor.std():.6f} / {tensor.min():.6f} / {tensor.max():.6f}\n")
            f.write(f"  - Non-zero elements: {torch.count_nonzero(tensor).item()} / {tensor.numel()}\n\n")
    
    print(f"✅ Analysis report saved to: {report_file}")
    
    # Verify that our patch hook worked as expected
    print("\n--- Verifying Patches ---")
    if 'block0_context_output' in model.extracted_activations:
        context_tensor = model.extracted_activations['block0_context_output']['tensor']
        if context_tensor.dim() >= 3:
            # Check the part of the tensor we tried to modify
            first_feature_slice = context_tensor[0, :, 0]
            is_zeroed = (first_feature_slice == 0.0).all().item()
            print(f"❓ Was 'zero_context_patch' successful? -> {is_zeroed}")
            if not is_zeroed:
                print(f"  -> Patch might not have worked. Mean of slice: {first_feature_slice.mean():.6f}")
    print("="*60)

def main():
    """Main inference process."""
    parser = argparse.ArgumentParser(prog="python3 test_hooks_inference.py", description="F5-TTS Inference with Hook Testing.")
    parser.add_argument("-c", "--config", type=str, default=str(files("f5_tts").joinpath("infer/examples/basic/basic.toml")))
    parser.add_argument("-m", "--model", type=str, help="Model name (e.g., F5TTS_v1_Base)")
    parser.add_argument("-r", "--ref_audio", type=str, help="Reference audio file.")
    parser.add_argument("-s", "--ref_text", type=str, help="Transcript for the reference audio.")
    parser.add_argument("-t", "--gen_text", type=str, help="Text to synthesize.")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output folder.")
    parser.add_argument("--enable_hooks", action="store_true", help="Enable hook functionality for testing.")
    args = parser.parse_args()

    config = tomli.load(open(args.config, "rb"))

    # Set parameters from args or config file
    model_name = args.model or config.get("model", "F5TTS_v1_Base")
    ref_audio = args.ref_audio or config.get("ref_audio", str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")))
    ref_text = args.ref_text if args.ref_text is not None else config.get("ref_text", "Some call me nature, others call me mother nature.")
    gen_text = args.gen_text or config.get("gen_text", "Hello world, this is a test of the hook system!")
    output_dir = args.output_dir or config.get("output_dir", "test_hooks_output")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- F5-TTS INFERENCE WITH HOOKS ---")
    print(f"‣ Model: {model_name}")
    print(f"‣ Reference Audio: {ref_audio}")
    print(f"‣ Generation Text: {gen_text}")
    print(f"‣ Output Directory: {output_dir}")
    print(f"‣ Hooks Enabled: {args.enable_hooks}")
    print("-" * 40)

    # --- Load Models ---
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_file = str(cached_path(f"hf://SWivid/F5-TTS/{model_name}/model_1250000.safetensors"))
    
    print(f"Loading model: {model_name}...")
    ema_model = load_model(model_cls, model_arc, ckpt_file, mel_spec_type="vocos", vocab_file="", device=device)

    # --- ‼️ IMPORTANT DIAGNOSTIC STEP ‼️ ---
    print("\n" + "="*60)
    print("INSPECTING MODEL ARCHITECTURE TO VERIFY HOOK PATHS...")
    print("="*60)
    for name, module in ema_model.named_modules():
        print(name)
    print("="*60 + "\n")
    print("👆 Check the names above (e.g., 'transformer_blocks.0.attn') and ensure they match the paths in setup_model_hooks().")


    # --- Setup Hooks if Enabled ---
    if args.enable_hooks:
        if not setup_model_hooks(ema_model):
            return # Stop if hook setup fails

    # --- Run Inference ---
    print(f"\n--- RUNNING INFERENCE ---")
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(ref_audio, ref_text)
    
    print("Generating audio...")
    audio_segment, final_sample_rate, _ = infer_process(
        ref_audio_processed, ref_text_processed, gen_text, ema_model, vocoder,
        mel_spec_type="vocos", target_rms=target_rms, cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef,
        speed=speed, fix_duration=fix_duration, device=device,
    )
    
    # --- Save Audio ---
    output_file = f"hooked_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wave_path = Path(output_dir) / output_file
    sf.write(wave_path, audio_segment, final_sample_rate)
    print(f"✅ Audio saved to: {wave_path}")
    
    # --- Analyze Hooks if Enabled ---
    if args.enable_hooks:
        save_hook_results(ema_model, output_dir)
        ema_model.remove_hooks()
        print("\n✅ Inference and analysis complete.")

if __name__ == "__main__":
    main()