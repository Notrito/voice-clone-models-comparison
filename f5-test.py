#!/usr/bin/env python3
"""
Prueba simple: Solo verificar que F5-TTS se puede importar
"""

def test_basic_imports():
    """Probar imports básicos"""
    print("🔍 Probando imports básicos...")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        
        import torchaudio  
        print(f"   ✅ TorchAudio: {torchaudio.__version__}")
        
        print("   📦 Importando F5-TTS...")
        import f5_tts
        print(f"   ✅ F5-TTS importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    if success:
        print("\n🎉 Imports funcionan correctamente!")
    else:
        print("\n❌ Hay problemas con los imports")