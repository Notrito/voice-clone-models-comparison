## 🎯 Quick Demo

Try it now without installing: **[Live Demo](https://huggingface.co/spaces/notrito/voice-clone-models-comparison)**

## 🔧 Run Locally

# F5-TTS Voice Cloning with Denoising Visualization

An interactive tool for voice cloning using F5-TTS that visualizes the denoising process in real-time, showing how the model transforms pure noise into natural speech step by step.

## Features

- **Zero-shot voice cloning**: Clone any voice with just 5-30 seconds of reference audio
- **Diffusion process visualization**: Observe all 32 denoising steps from noise to clean audio
- **Interactive interface**: Control which intermediate step to listen to with a slider
- **Multiple languages**: Official support for English and Chinese (other languages experimental)
- **CPU processing**: No GPU required to run

## Live Demo

Try the application without installing anything: [Demo on Hugging Face Spaces](https://huggingface.co/spaces/notrito/voice-clone-models-comparison)

## Prerequisites

- Python 3.10 or higher
- 4GB+ RAM
- Clean reference audio (no background noise)

## Installation

Start the application:

python app.py

Open your browser at http://localhost:7860
Steps to generate voice:

Upload reference audio (5-30 seconds)
Write the exact transcription of the audio
Write the text you want to generate with that voice
Click "Generate with Step Capture"
Use the slider to explore intermediate steps of the process



What does each step do?
The F5-TTS model uses 32 diffusion steps to generate audio:

Step 0: Pure random noise (starting point)
Step 12: First speech structures emerge (very distorted)
Step 20: Distinguishable voice patterns (with artifacts)
Step 26: Almost clean audio (minor imperfections)
Step 32: Completely clean and natural audio (final result)

Tech Stack

F5-TTS: Text-to-speech model with flow matching
Gradio: Framework for web interface
PyTorch: Deep learning backend
Vocos: Vocoder for converting mel-spectrograms to audio
torchaudio: Audio processing


Tips for Better Results

Clean audio: No background noise, music, or echo
Optimal duration: 5-30 seconds of reference audio
Exact transcription: Must match the audio exactly
Clear speech: Constant volume and clear pronunciation
Language: For best results, use English or Chinese

Known Limitations

Base model is trained primarily on English and Chinese
Other languages may work but with variable quality
CPU processing may take 30-60 seconds per generation
Requires initial model download (~600MB)

Credits

F5-TTS: SWivid - Shanghai Jiao Tong University
Original paper: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
Vocos Vocoder: Charactr

License
This project uses the F5-TTS model which is under MIT License. Pre-trained models are under CC-BY-NC license due to the Emilia dataset used in training.
Contributing
Contributions are welcome. Please:


Contact
Noel Triguero - LinkedIn
