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
SUPPORTED_LANGUAGES = ["es", "en"]
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB

# Variables globales para el modelo (se cargan una vez)
model = None
vocoder = None
model_loaded = False

def load_models():
    """Cargar F5-TTS y vocoder (solo una vez al iniciar)"""
    global model, vocoder, model_loaded
    
    if model_loaded:
        return True
    
    try:
        print("⏳ Cargando F5-TTS y vocoder...")
        print("=" * 50)
        
        # Cargar vocoder primero
        print("📥 Cargando vocoder Vocos...")
        vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=False,
            device="cpu"
        )
        print("✅ Vocoder cargado correctamente")
        
        # Configuración del modelo (copiado del código oficial)
        print("\n📥 Cargando modelo F5-TTS v1 Base...")
        
        ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
        model_cfg = dict(
            dim=1024, 
            depth=22, 
            heads=16, 
            ff_mult=2, 
            text_dim=512, 
            conv_layers=4
        )
        
        # Cargar modelo usando la misma función que el código oficial
        model = load_model(
            DiT,
            model_cfg,
            ckpt_path
        )
        print("✅ Modelo F5-TTS cargado correctamente")
        
        model_loaded = True
        print("\n" + "=" * 50)
        print("✅ Todos los modelos cargados exitosamente")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO cargando modelos:")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Mensaje: {str(e)}")
        import traceback
        print("\nStack trace completo:")
        traceback.print_exc()
        print("=" * 50)
        return False

def validate_audio(audio_file):
    """Validar archivo de audio"""
    if audio_file is None:
        return False, "Por favor, sube un archivo de audio"
    
    try:
        file_size = os.path.getsize(audio_file)
        if file_size > MAX_AUDIO_SIZE:
            return False, f"Archivo muy grande. Máximo 10MB"
        return True, "Audio válido"
    except Exception as e:
        return False, f"Error validando audio: {e}"

def generate_voice(reference_audio, ref_text, gen_text, language):
    """Generar voz con F5-TTS"""
    
    # Validar entrada
    is_valid, msg = validate_audio(reference_audio)
    if not is_valid:
        return None, f"❌ {msg}", ""
    
    if not ref_text or not ref_text.strip():
        return None, "❌ Debes escribir la transcripción del audio de referencia", ""
    
    if not gen_text or not gen_text.strip():
        return None, "❌ Debes escribir el texto a generar", ""
    
    # Verificar que los modelos estén cargados
    if not model_loaded:
        success = load_models()
        if not success:
            return None, "❌ Error cargando modelos. Intenta recargar la página.", ""
    
    try:
        start_time = time.time()
           
        print(f"🎤 Generando audio...")
        print(f"   Ref text: {ref_text[:50]}...")
        print(f"   Gen text: {gen_text[:50]}...")
        
        # Preprocesar audio de referencia
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            reference_audio, 
            ref_text
        )
        
        # Procesar con F5-TTS (igual que el código oficial)
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
        
        # result debería ser el audio generado
        output_path = "generated_audio.wav"
        
        success_msg = f"✅ Audio generado exitosamente"
        time_msg = f"⏱️ Tiempo: {processing_time:.2f}s"
        
        return (final_sample_rate, final_wave), success_msg, time_msg
        
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}", ""

def generate_voice_with_steps(reference_audio, ref_text, gen_text, language):
    """Generar voz capturando pasos intermedios del denoising"""
    
    # Validar entrada
    is_valid, msg = validate_audio(reference_audio)
    if not is_valid:
        return None, None, f"❌ {msg}"
    
    if not ref_text or not ref_text.strip():
        return None, None, "❌ Debes escribir la transcripción del audio de referencia"
    
    if not gen_text or not gen_text.strip():
        return None, None, "❌ Debes escribir el texto a generar"
    
    # Verificar que los modelos estén cargados
    if not model_loaded:
        success = load_models()
        if not success:
            return None, None, "❌ Error cargando modelos"
    
    try:       
        print("🔬 Generando con captura de pasos intermedios...")
        
        # Preprocesar
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            reference_audio, 
            ref_text
        )
        
        # Cargar y procesar audio
        audio, sr = torchaudio.load(ref_audio_processed)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resamplear si es necesario
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        
        audio = audio.to("cpu")
        
        # Preparar texto
        text_list = [ref_text_processed + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)
        
        # Calcular duración
        ref_audio_len = audio.shape[-1] // 256  # hop_length
        ref_text_len = len(ref_text_processed.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)
        
        # Generar CON trajectory
        print("Llamando a model.sample() con captura de trajectory...")
        with torch.inference_mode():
            generated_mel, trajectory = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )
        
        print(f"Trajectory capturado - Shape: {trajectory.shape}")
        
        # Extraer pasos específicos para mostrar
        steps_to_extract = [0, 8, 16, 24, 32]
        step_audios = []
        
        for step_idx in steps_to_extract:
            print(f"Procesando paso {step_idx}/32...")
            mel_at_step = trajectory[step_idx]
            
            # Recortar parte de referencia y permutar
            mel_generated = mel_at_step[:, ref_audio_len:, :]
            mel_generated = mel_generated.permute(0, 2, 1)
            
            # Convertir a audio con vocoder
            audio_at_step = vocoder.decode(mel_generated)
            audio_np = audio_at_step.squeeze().cpu().numpy()
            
            step_audios.append((24000, audio_np))
        
        # El último paso es el audio final
        final_audio = step_audios[-1]
        
        print("✅ Generación con pasos completada")
        
        # Retornar: audio final, lista de pasos, mensaje
        return final_audio, step_audios, f"✅ Generado con captura de {len(steps_to_extract)} pasos intermedios"
        
    except Exception as e:
        print(f"❌ Error en generación con pasos: {e}")
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
        gr.Markdown("Clona cualquier voz con solo 5-30 segundos de audio de referencia")
        
        with gr.Tabs():
            # Tab 1: Generación básica
            with gr.Tab("Generación Básica"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 📁 Entrada")
                        
                        reference_audio = gr.Audio(
                            label="Audio de Referencia (5-30 segundos)",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        ref_text = gr.Textbox(
                            label="Transcripción del Audio de Referencia",
                            placeholder="Escribe exactamente lo que dice el audio de referencia...",
                            lines=2,
                            info="Importante: Debe coincidir con lo que dice el audio"
                        )
                        
                        gen_text = gr.Textbox(
                            label="Texto a Generar",
                            placeholder="Escribe el texto que quieres que diga con la voz clonada...",
                            lines=3
                        )
                        
                        language = gr.Dropdown(
                            choices=SUPPORTED_LANGUAGES,
                            value="es",
                            label="Idioma",
                            info="Idioma del texto a generar"
                        )
                        
                        generate_btn = gr.Button("🚀 Generar Voz", variant="primary", size="lg")
                
                with gr.Row():
                    status_msg = gr.Textbox(label="Estado", interactive=False, show_label=False)
                
                with gr.Row():
                    time_msg = gr.Textbox(label="Tiempo de Procesamiento", interactive=False)
                
                with gr.Row():
                    output_audio = gr.Audio(label="🔊 Audio Generado", type="filepath")
                
                generate_btn.click(
                    fn=generate_voice,
                    inputs=[reference_audio, ref_text, gen_text, language],
                    outputs=[output_audio, status_msg, time_msg]
                )
            
            Tab 2: Visualización del proceso de denoising
            with gr.Tab("Visualización del Denoising"):
                gr.Markdown("""
                ## 🔬 Visualización del Proceso de Denoising
                
                Esta sección te permite ver cómo el modelo transforma ruido puro en audio limpio paso a paso.
                El modelo F5-TTS usa 32 pasos de "denoising" para generar el audio final.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Entrada")
                        
                        ref_audio_steps = gr.Audio(
                            label="Audio de Referencia",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        ref_text_steps = gr.Textbox(
                            label="Transcripción",
                            lines=2
                        )
                        
                        gen_text_steps = gr.Textbox(
                            label="Texto a Generar",
                            lines=3
                        )
                        
                        language_steps = gr.Dropdown(
                            choices=SUPPORTED_LANGUAGES,
                            value="es",
                            label="Idioma"
                        )
                        
                        generate_steps_btn = gr.Button(
                            "🔬 Generar con Captura de Pasos", 
                            variant="primary"
                        )
                
                with gr.Row():
                    status_steps = gr.Textbox(label="Estado", interactive=False)
                
                with gr.Row():
                    gr.Markdown("### Audio Final")
                    final_audio_output = gr.Audio(label="Resultado Final", type="numpy")
                
                gr.Markdown("### Pasos Intermedios del Denoising")
                
                with gr.Row():
                    step_slider = gr.Slider(
                        minimum=0,
                        maximum=4,
                        value=4,
                        step=1,
                        label="Seleccionar Paso",
                        info="0=Ruido inicial, 1=Paso 8, 2=Paso 16, 3=Paso 24, 4=Paso 32 (final)"
                    )
                
                with gr.Row():
                    step_audio = gr.Audio(
                        label="Audio en el Paso Seleccionado",
                        type="numpy"
                    )
                
                # Estado oculto para guardar todos los pasos
                all_steps_state = gr.State(value=None)
                
                def update_step_audio(step_index, all_steps):
                    if all_steps is None:
                        return None
                    return all_steps[int(step_index)]
                
                # Generar y guardar pasos
                def process_with_steps(ref_audio, ref_text, gen_text, lang):
                    final, steps, status = generate_voice_with_steps(
                        ref_audio, ref_text, gen_text, lang
                    )
                    # Retornar: audio final, todos los pasos (para state), último paso para mostrar, estado
                    return final, steps, steps[-1] if steps else None, status
                
                generate_steps_btn.click(
                    fn=process_with_steps,
                    inputs=[ref_audio_steps, ref_text_steps, gen_text_steps, language_steps],
                    outputs=[final_audio_output, all_steps_state, step_audio, status_steps]
                )
                
                step_slider.change(
                    fn=update_step_audio,
                    inputs=[step_slider, all_steps_state],
                    outputs=[step_audio]
                )
                
                gr.Markdown("""
                ### 📊 Explicación de los Pasos
                
                - **Paso 0 (Ruido)**: Ruido aleatorio puro - el punto de partida
                - **Paso 8**: Primeras estructuras emergen, muy distorsionado
                - **Paso 16**: Se distinguen patrones de habla, aún con artefactos
                - **Paso 24**: Audio casi limpio, algunas imperfecciones
                - **Paso 32 (Final)**: Audio completamente limpio y natural
                
                Este proceso se llama "diffusion" - el modelo aprende a "limpiar" ruido gradualmente.
                """)
        
        gr.Markdown("""
        ## 💡 Consejos para Mejores Resultados
        
        - **Audio limpio:** Sin ruido de fondo, música o eco
        - **Duración:** 5-30 segundos es ideal
        - **Transcripción exacta:** La transcripción debe coincidir exactamente con el audio
        - **Habla clara:** Volumen constante y pronunciación clara
        - **Idioma:** El audio de referencia y el texto pueden estar en idiomas diferentes
        
        ## 🔧 Información Técnica
        
        - **Modelo:** F5-TTS (Flow Matching Text-to-Speech)
        - **Vocoder:** Vocos
        - **Dispositivo:** CPU (puede tardar ~30-60 segundos)
        """)
    
    return demo

if __name__ == "__main__":
    # Pre-cargar modelos al iniciar (opcional, mejora primera experiencia)
    print("🚀 Iniciando F5-TTS Voice Cloning App")
    print("=" * 50)
    
    # Comentar la siguiente línea si quieres carga bajo demanda
    # load_models()
    
    demo = create_interface()
    demo.launch()