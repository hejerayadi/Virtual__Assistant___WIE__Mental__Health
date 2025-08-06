# Virtual__Assistant___WIE__Mental__Health

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hejerayadi/Virtual__Assistant___WIE__Mental__Health 
   cd "virtual assistant mental health"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   Open your browser and go to `http://127.0.0.1:5000/`

## Project Description

This project is a virtual assistant designed to support mental health by analyzing user input through text, voice, and video. It leverages deep learning and large language models to detect emotions and provide supportive responses. The assistant can process spoken, written, and visual cues to better understand the user's emotional state and offer appropriate feedback or conversation.

## Main Functionalities
- **Text Emotion Detection:** Analyze user-provided text to detect emotions and provide a supportive response using an LLM (Large Language Model).
- **Voice Emotion Detection & Transcription:** Accept audio input, transcribe it using Whisper, detect the speaker's emotion, and generate a response (with optional text-to-speech output).
- **Facial Emotion Detection:** Analyze webcam or image input to detect facial emotions in real time.
- **Conversational Support:** Generate context-aware, supportive responses to user input using an LLM.
- **Text-to-Speech:** Convert generated responses to speech for audio playback.

## Requirements

- Python 3.8+
- A working microphone (for voice input)
- A webcam (for facial emotion detection)

## Model Files

The project expects the following files in the `Emotion_Detection_CNN/` directory:
- `model.h5` (pre-trained emotion detection model)
- `haarcascade_frontalface_default.xml` (for face detection)

These files should already be present in the repository.

## Notes
- For voice and video functionalities, ensure your device has a working microphone and webcam.
- The app uses pre-trained models for emotion detection and Whisper for speech-to-text.
- All processing is done locally; no data is sent to external servers by default.
