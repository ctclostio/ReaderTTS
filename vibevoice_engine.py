import sys
import os
# Add VibeVoice_new to path to locate the 'vibevoice' package
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "VibeVoice_new"))

import torch
import numpy as np
import random
from transformers.utils import logging
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

logging.set_verbosity_error() # Reduce spam

class VibeVoiceClient:
    def __init__(self, model_name="microsoft/VibeVoice-1.5b"):
        # Select precision based on model size
        # 7B models require float16 to fit in VRAM (24GB). 1.5B usually fits in float32.
        if "7b" in model_name.lower():
            print(f"Detected 7B model ({model_name}). Switching to float16 to save VRAM.")
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing VIBEVOICE Engine...")
        print("APEX FusedRMSNorm not available, using native implementation")
        
        print(f"Loading VibeVoice model from {model_name} on {self.device} ({self.dtype})...")
        
        try:
            self.processor = VibeVoiceProcessor.from_pretrained(model_name)
            
            # Note: Inference class might be needed for specific generation methods
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_name, 
                torch_dtype=self.dtype
            ).to(self.device)
            
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10) # Default to reasonable speed
            print("VibeVoice model loaded successfully.")
            
        except OSError as e:
            print(f"Error loading model: {e}")
            print("If using 7B, ensure you have downloaded it or provided the correct HuggingFace ID (e.g. aoi-ot/VibeVoice-7B).")
            raise e
        except Exception as e:
            print(f"Failed to load VibeVoice: {e}")
            raise e

    def generate_audio(self, text_chunks, ref_audios, ref_texts, **kwargs):
        """
        Generator that yields audio chunks.
        
        Args:
            text_chunks (list): List of text strings to generate.
            ref_audios (list): List of paths to reference audio files.
            ref_texts (dict or str): Reference text mapping or single string.
            **kwargs: Additional generation parameters (speed, etc - currently mostly unused for VibeVoice basic impl)
        """
        
        # Configuration parameters
        cfg_scale = kwargs.get('cfg_strength', 1.5)
        num_steps = kwargs.get('nfe_step', 30)
        self.model.set_ddpm_inference_steps(num_steps=num_steps)
        print(f"VibeVoice config: steps={num_steps}, cfg={cfg_scale}")
        
        # Load and concatenate all reference audios to create a stable, multi-clip voice prompt
        # This addresses inconsistency by using ALL available data for every chunk
        combined_ref_audio = []
        import librosa
        import numpy as np
        
        print(f"Loading and combining {len(ref_audios)} reference clips for stable voice prompt...")
        
        # Memory Optimization: Full dataset (2000+ files) causes OOM on 24GB VRAM.
        # We need to limit the prompt length.
        # User requested 100s limit. This is likely near the upper bound for 24GB VRAM/Context Window.
        target_duration = 100.0 
        current_duration = 0.0
        
        # Shuffle to get a variety of "Kratos" tones each time
        import random
        random.shuffle(ref_audios)
        
        torch.cuda.empty_cache() # Clear any residual memory
        
        selected_clips = []
        
        print(f"Selecting random subset of clips to reach ~{target_duration}s...")
        for ref_path in ref_audios:
            if current_duration >= target_duration:
                break
                
            try:
                target_sr = getattr(self.processor.audio_processor, 'sampling_rate', 24000)
                audio, _ = librosa.load(ref_path, sr=target_sr)
                combined_ref_audio.append(audio)
                current_duration += len(audio) / target_sr
                selected_clips.append(ref_path)
            except Exception as e:
                print(f"Error loading ref {ref_path}: {e}")
        
        if combined_ref_audio:
            final_ref_audio = np.concatenate(combined_ref_audio)
            print(f"Final voice prompt length: {len(final_ref_audio)/target_sr:.2f}s (from {len(combined_ref_audio)} clips)")
            print("\n--- Selected Clips ---")
            for clip in selected_clips:
                print(f"USED: {os.path.basename(clip)}")
            print("----------------------\n")
            
            # Optional: Copy to Known Good folder if env var is set or just hardcoded for this request
            # For now, just printing them allows the user to know. 
            # We can also auto-copy them to "SampleAudio/Trump_KnownGood" if we want to be proactive.
            known_good_dir = os.path.join(os.path.dirname(ref_audios[0]), "Known_Good_Candidates")
            if not os.path.exists(known_good_dir):
                os.makedirs(known_good_dir, exist_ok=True)
                
            import shutil
            for clip in selected_clips:
                try:
                    shutil.copy(clip, known_good_dir)
                except shutil.SameFileError:
                    pass # Already there
                except Exception as e:
                    print(f"Warning: Could not copy {clip} to known good: {e}")
            print(f"Copied selected clips to: {known_good_dir}")
                
        else:
            final_ref_audio = None
            
        for i, chunk in enumerate(text_chunks):
            # Clean chunk to remove newlines, as VibeVoice processor splits by line and expects "Speaker X:" prefix on EACH line
            clean_chunk = chunk.replace('\n', ' ').strip()
            
            # VibeVoice expects "Speaker 1: <text>" format
            formatted_text = f"Speaker 1: {clean_chunk}"
            
            try:
                inputs = self.processor(
                    text=[formatted_text],
                    voice_samples=[[final_ref_audio]], 
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                
                # Move inputs to device
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(self.device)
                
                print(f"Generating chunk {i+1}/{len(text_chunks)} using combined 'Master Voice' prompt...")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False}, 
                    )
                
                # Extract audio
                if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                    audio_tensor = outputs.speech_outputs[0]
                    # Convert to numpy
                    audio_np = audio_tensor.cpu().numpy()
                    
                    # Check dimensions - usually (samples,) or (1, samples)
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                        
                    yield audio_np, 24000 # Yield audio and sample rate (VibeVoice usually 24k)
                else:
                    print(f"Warning: No audio generated for chunk {i+1}")
                    yield None, 24000

            except Exception as e:
                print(f"Error generating chunk {i+1}: {e}")
                yield None, 24000
