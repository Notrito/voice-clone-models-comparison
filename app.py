import gradio as gr
import time
import os
from pathlib import Path

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
        
        import json
        from cached_path import cached_path
        from f5_tts.infer.utils_infer import load_model, load_vocoder
        from f5_tts.model import DiT
        
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
        
        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
        
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
        
        return output_path, success_msg, time_msg
        
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}", ""

# Crear interfaz Gradio
def create_interface():
    with gr.Blocks(
        title="F5-TTS Voice Cloning",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("# 🎤 F5-TTS Voice Cloning")
        gr.Markdown("Clona cualquier voz con solo 5-30 segundos de audio de referencia")
        
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
        
        # Event handlers
        generate_btn.click(
            fn=generate_voice,
            inputs=[reference_audio, ref_text, gen_text, language],
            outputs=[output_audio, status_msg, time_msg]
        )
        
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