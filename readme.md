# Transcribe Panel

Transcribe Panel is an interface for local and cloud transcription models for comparing their output speed, quality and pricing.

Try it here: https://landonikko.github.io/Transcribe-Panel

The hosted version is a local interface that works by using your own API keys and caches the data in your browser.

For testing local models, see the Python Setup.

![Transcribe Panel Screenshot](https://i.imgur.com/WIFcH6T.jpeg)

## Features

**Local Models** (OpenAI Whisper)
- Tiny, Base, Small, Medium, Turbo, Large

**Cloud Models** (Bring Your Own API Key)
- **OpenAI**: Whisper-1, GPT-4o, GPT-4o Mini
   - 4o supports custom prompting for translations and localization
- **Deepgram**: Nova-3
   - With AI summarization & topic detection
- **Gladia**: Gladia v2
- **ElevenLabs**: Scribe-v1, Scribe-v1-Experimental

**Smart Features**
- **Example Files**: Multi-language samples (EN, FR, JP)
- **Batch Processing**: Queue multiple files
- **Media Player**: Audio/video preview with waveform visualization
- **Live Sync**: Transcript segments are highlighted by audio position
- **Multiple Formats**: SRT, VTT, TSV, plain text, numbered segments
- **Transcript Editing**: Edit transcripts
- **Upload/Import**: Import existing transcript files
- **Cost Calculator**: API pricing estimates per file
- **UI Themes**: E Ink, Nova, Minty, Toffee
- **Statistics**: Word count, character count, processing time

## API Keys (Cloud Models Only)

All cloud providers offer free starting credits, with some offering monthly refills:

- [OpenAI](https://platform.openai.com/api-keys)
- [Deepgram](https://console.deepgram.com/signup)
- [Gladia](https://app.gladia.io)
- [ElevenLabs](https://elevenlabs.io)

*API keys can be stored in your browser the same way your passwords are saved.*

## Python Setup (For Local Models)

### Requirements
- Browser, Python 3.8+, (PyTorch [pytorch.org](https://pytorch.org)), FFmpeg

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LandoNikko/Transcribe-Panel.git
   cd Transcribe-Panel
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for media processing)
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

4. **Run the application**
   ```bash
   python Launch_Transcribe_Panel.py
   ```
   Open `http://localhost:5000` in your browser

## Tips

- **Local models**: No internet required, complete privacy. Relies on your hardware (GPU vram), so transcriptions can be slow.
- **Cloud models**: Fast, advanced features, requires API keys. Costs, but very cheap. Can be used heavily with just free credits.
- **File support**: Most audio/video formats (MP3, WAV, MP4, etc.)
- **Batch mode**: Select multiple files and process them sequentially.
- **Export options**: Copy to clipboard or download in various formats.

## License

LGPL-2.1 License - see [LICENSE](LICENSE) file.
