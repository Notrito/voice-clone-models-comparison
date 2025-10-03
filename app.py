import gradio as gr
import time
import os
from pathlib import Path
import json
from cached_path import cached_path
from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
import torch
import torchaudio
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, convert_char_to_pinyin
     

# Configuración
MODEL_NAME = "F5-TTS"
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "zh"]
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB

# Variables globales para el modelo (se cargan una vez)
model = None
vocoder = None
model_loaded = False

def load_models():
    """Load F5-TTS and vocoder (only once at startup)"""
    global model, vocoder, model_loaded
    
    if model_loaded:
        return True
    
    try:
        print("⏳ Loading F5-TTS and vocoder...")
        print("=" * 50)
        
        # Load vocoder first
        print("🔥 Loading Vocos vocoder...")
        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            device="cpu"
        )
        print("✅ Vocoder loaded successfully")
        
        # Model configuration (copied from official code)
        print("\n🔥 Loading F5-TTS v1 Base model...")
        
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
        model_cfg = dict(
            dim=1024, 
            depth=22, 
            heads=16, 
            ff_mult=2, 
            text_dim=512, 
            conv_layers=4
        )
        
        # Load model using the same function as the official code
        model = load_model(
            DiT,
            model_cfg,
            ckpt_path
        )
        print("✅ F5-TTS model loaded successfully")
        
        model_loaded = True
        print("\n" + "=" * 50)
        print("✅ All models loaded successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR loading models:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        print("\nFull stack trace:")
        traceback.print_exc()
        print("=" * 50)
        return False

def validate_audio(audio_file):
    """Validate audio file"""
    if audio_file is None:
        return False, "Please upload an audio file"
    
    try:
        file_size = os.path.getsize(audio_file)
        if file_size > MAX_AUDIO_SIZE:
            return False, f"File too large. Maximum 10MB"
        return True, "Valid audio"
    except Exception as e:
        return False, f"Error validating audio: {e}"

def generate_voice(reference_audio, ref_text, gen_text):
    """Generate voice with F5-TTS"""
    
    # Validate input
    is_valid, msg = validate_audio(reference_audio)
    if not is_valid:
        return None, f"❌ {msg}", ""
    
    if not ref_text or not ref_text.strip():
        return None, "❌ You must write the transcription of the reference audio", ""
    
    if not gen_text or not gen_text.strip():
        return None, "❌ You must write the text to generate", ""
    
    # Check that models are loaded
    if not model_loaded:
        success = load_models()
        if not success:
            return None, "❌ Error loading models. Try reloading the page.", ""
    
    try:
        start_time = time.time()
           
        print(f"🎤 Generating audio...")
        print(f"   Ref text: {ref_text[:50]}...")
        print(f"   Gen text: {gen_text[:50]}...")
        
        # Preprocess reference audio
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            reference_audio, 
            ref_text
        )
        
        # Process with F5-TTS (same as official code)
        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=gen_text,
            model_obj=model,
            vocoder=vocoder,
            device="cpu"
        )
        end_time = time.time()
        processing_time = end_time - start_time
        
        # result should be the generated audio
        output_path = "generated_audio.wav"
        
        success_msg = f"✅ Audio generated successfully"
        time_msg = f"⏱️ Time: {processing_time:.2f}s"
        
        return (final_sample_rate, final_wave), success_msg, time_msg
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}", ""

def generate_voice_with_steps(reference_audio, ref_text, gen_text):
    """Generate voice capturing intermediate denoising steps"""
    
    # Validate input
    is_valid, msg = validate_audio(reference_audio)
    if not is_valid:
        return None, None, f"❌ {msg}"
    
    if not ref_text or not ref_text.strip():
        return None, None, "❌ You must write the transcription of the reference audio"
    
    if not gen_text or not gen_text.strip():
        return None, None, "❌ You must write the text to generate"
    
    # Check that models are loaded
    if not model_loaded:
        success = load_models()
        if not success:
            return None, None, "❌ Error loading models"
    
    try:       
        print("🔬 Generating with intermediate step capture...")
        
        # Preprocess
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            reference_audio, 
            ref_text
        )
        
        # Load and process audio
        audio, sr = torchaudio.load(ref_audio_processed)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        
        audio = audio.to("cpu")
        
        # Prepare text
        text_list = [ref_text_processed + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        
        # Calculate duration
        ref_audio_len = audio.shape[-1] // 256  # hop_length
        ref_text_len = len(ref_text_processed.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
        
        # Generate WITH trajectory
        print("Calling model.sample() with trajectory capture...")
        with torch.inference_mode():
            generated_mel, trajectory = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )
        
        print(f"Trajectory captured - Shape: {trajectory.shape}")
        
        # Extract specific steps to display
        steps_to_extract = [0, 8, 16, 24, 32]
        step_audios = []
        
        for step_idx in steps_to_extract:
            print(f"Processing step {step_idx}/32...")
            mel_at_step = trajectory[step_idx]
            
            # Crop reference part and permute
            mel_generated = mel_at_step[:, ref_audio_len:, :]
            mel_generated = mel_generated.permute(0, 2, 1)
            
            # Convert to audio with vocoder
            audio_at_step = vocoder.decode(mel_generated)
            audio_np = audio_at_step.squeeze().cpu().numpy()
            
            step_audios.append((24000, audio_np))
        
        # The last step is the final audio
        final_audio = step_audios[-1]
        
        print("✅ Generation with steps completed")
        
        # Return: final audio, list of steps, message
        return final_audio, step_audios, f"✅ Generated with capture of {len(steps_to_extract)} intermediate steps"
        
    except Exception as e:
        print(f"❌ Error in generation with steps: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"    
# Crear interfaz Gradio

def create_interface():
    with gr.Blocks(
        title="F5-TTS Voice Cloning",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("# 🎤 F5-TTS Voice Cloning")
        gr.Markdown("Clone any voice with just 5-30 seconds of reference audio")
        gr.Markdown("Developed by Noel Triguero. Model by SWivid")
        gr.Markdown("---")

        gr.Markdown("""
        ## 🔬 Denoising Process Visualization
        
        This section lets you see how the model transforms pure noise into clean audio step by step.
        The F5-TTS model uses 32 "denoising" steps to generate the final audio.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                ref_audio_steps = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                ref_text_steps = gr.Textbox(
                    label="Transcription",
                    lines=2
                )
                
                gen_text_steps = gr.Textbox(
                    label="Text to Generate",
                    lines=3
                )
                
                generate_steps_btn = gr.Button(
                    "🔬 Generate with Step Capture", 
                    variant="primary"
                )
        
        with gr.Row():
            status_steps = gr.Textbox(label="Status", interactive=False)
        
        with gr.Row():
            gr.Markdown("### Final Audio ")
            final_audio_output = gr.Audio(label="Final Result", type="numpy")
        
        gr.Markdown("### Intermediate Denoising Steps")
        
        with gr.Row():
            step_slider = gr.Slider(
                minimum=0,
                maximum=4,
                value=4,
                step=1,
                label="Select Step",
                info="0=Initial noise, 1=Step 8, 2=Step 16, 3=Step 24, 4=Step 32 (final)"
            )
        
        with gr.Row():
            step_audio = gr.Audio(
                label="Audio at Selected Step",
                type="numpy"
            )
        
        # Hiden state to store all steps
        all_steps_state = gr.State(value=None)
        
        def update_step_audio(step_index, all_steps):
            if all_steps is None:
                return None
            return all_steps[int(step_index)]
        
        # Generate with steps and store all steps in state
        def process_with_steps(ref_audio, ref_text, gen_text):
            final, steps, status = generate_voice_with_steps(
                ref_audio, ref_text, gen_text
            )
            # Only return the last step audio for the slider
            if steps:
                return final, steps, steps[-1], status
            else:
                return None, None, None, status
        
        generate_steps_btn.click(
            fn=process_with_steps,
            inputs=[ref_audio_steps, ref_text_steps, gen_text_steps],
            outputs=[final_audio_output, all_steps_state, step_audio, status_steps]
        )
        
        step_slider.change(
            fn=update_step_audio,
            inputs=[step_slider, all_steps_state],
            outputs=[step_audio]
        )
                
        gr.Markdown("""
        ### 📊 Step Explanation
        
        - **Step 0 (Noise)**: Pure random noise - the starting point
        - **Step 8**: First structures emerge, very distorted
        - **Step 16**: Speech patterns distinguishable, still with artifacts
        - **Step 24**: Almost clean audio, some imperfections
        - **Step 32 (Final)**: Completely clean and natural audio
        
        This process is called "diffusion" - the model learns to "clean" noise gradually.
        """)
        gr.Markdown("""
        ## 💡 Tips for Better Results
        
        - **Clean audio:** No background noise, music or echo
        - **Duration:** 5-30 seconds is ideal
        - **Exact transcription:** The transcription must match the audio exactly
        - **Clear speech:** Constant volume and clear pronunciation
        
        ## 🔧 Technical Information
        
        - **Model:** F5-TTS (Flow Matching Text-to-Speech)
        - **Vocoder:** Vocos
        - **Device:** CPU (may take ~30-60 seconds)
        """)
    
    return demo

if __name__ == "__main__":
    # Pre-load models at startup (optional, improves first experience)
    print("🚀 Starting F5-TTS Voice Cloning App")
    print("=" * 50)
    
    # Comment the following line if you want on-demand loading
    # load_models()
    
    demo = create_interface()
    demo.launch()