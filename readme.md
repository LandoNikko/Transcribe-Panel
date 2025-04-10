# Kaira Transcribe Panel

Kaira Transcribe Panel is a web app for local and cloud transcription (STT: speech-to-text) models. It's a privacy-conscious tool for transcribing audio/video files using OpenAI's open-source Whisper models, as well as offering cloud options (OpenAI, Deepgram, Gladia) where you use your own API key. Compare each model's output speed, quality and pricing. Think of this like Automatic1111, but made for transcriptions.

![Screenshot Placeholder - Add one here]

---

## Table of Contents

- [Features Overview](#features-overview)
- [Installation](#installation)
- [Launching the App](#launching-the-app)
- [Cloud Models](#cloud-models)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License & Credits](#license--credits)

---

## Features Overview

- **Local Models** (via `openai-whisper`):
  - Tiny, Base, Small, Medium, Turbo and Large.
- **Cloud Models (BYOK — Bring Your Own Key):**
  - OpenAI Whisper-1
  - Deepgram Nova-3 *(Summarization & Topic Detection supported)*
  - Gladia *(English-only for now)*
- **Transcript Outputs:** Paragraphs, Segments, SRT, VTT and TSV.
- **Media Sync:** Built-in audio/video player with waveform, real-time paragraph highlighting synced to playback and click-to-seek transcript.
- **Personalization:** From light mode to dark modes and cute (E Ink, Nova, Minty, Toffee).
- **Queue Support**
  - Add/remove and transcribe multiple files.
- **Price Calculator:** Estimates API costs for cloud models.
- **Convenience:** Copy/download results, notifications, word/character stats.

---

## Installation

### 1. Clone and Enter the Project Directory

```bash
git clone https://github.com/kairauser/KairaTranscribePanel.git
cd KairaTranscribePanel
```

### 2. Install Python + Dependencies

- Use **Python 3.8+**
- Install **PyTorch** for your system: [https://pytorch.org](https://pytorch.org)

### ffmpeg

To enable audio/video processing, install **ffmpeg**:

- **macOS:**
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt install ffmpeg
  ```
- **Windows:**
  Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system PATH.

### Other Python dependencies

```bash
pip install -r requirements.txt
```

> Sample `requirements.txt`:
> ```
> Flask
> Flask-SocketIO
> openai-whisper
> openai
> requests
> torch
> ```

### Project Structure

```
Kaira_Transcribe_Panel/
├── temp/                    # Processing/transcribing files are temporarily stored here
├── Kaira_Transcribe_Panel.html  # Local interface
├── Launch_Kaira.py         # Launcher script for the app
├── readme.md               # Project documentation
├── requirements.txt        # Python dependencies
```

---

## Launching the App

### 1. Run

```bash
python Launch_Kaira.py
```

### 2. Open the Web App

Visit `http://localhost:5000` in your browser.

---

## Cloud Models

To transcribe with **Whisper-1**, **Deepgram**, or **Gladia**:

- Paste your API key when prompted in the UI.
- API keys are stored **only** in your **browser's localStorage**.
- You can remove them anytime by clearing your browser storage.

### Where to Get API Keys

- [OpenAI](https://platform.openai.com/account/api-keys)
- [Deepgram](https://console.deepgram.com/signup)
- [Gladia](https://gladia.io)

---

## Roadmap

- Translation -tab with different options (API provider's offers and LLM workflow integration).
- New transcription models, local and cloud.
- Explore in-browser-only Whisper execution (e.g., ONNX, Transformers.js).
- Accessibility (ARIA-compliant components).
- Eliminate the external CDN requirement to bundle WaveSurfer.js locally (so you don't need an internet connection to see the audio player).
- Modular plugin-style cloud support (like A1111 extensions) to keep local version minimal by default.

---

## Contributing

This project is a personal hobby project and under active development. While it's not a full-fledged product, contributions are very welcome if you find the tool useful or want to help it evolve.

We follow a simple and open workflow:

1. **Fork the repository.**
2. **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix`.
3. **Make your changes.** Try to keep things consistent with the UI/UX.
4. **Test your changes thoroughly.**
5. **Commit your changes:** `git commit -m "Add concise description of changes"`.
6. **Push to your branch:** `git push origin feature/your-feature-name`.
7. **Create a Pull Request** on the original repository.

> This project uses the Fork + Pull Request workflow. Anyone is welcome to fork the repo, make changes and submit a PR.

Please also feel free to open **Issues** for bug reports, feature requests, feedback or share how you use it.

---

## License & Credits

**License:** MIT — see the [`LICENSE`](LICENSE) file.

**Credits:**

- [OpenAI Whisper](https://github.com/openai/whisper)
- [WaveSurfer.js](https://wavesurfer-js.org/)
- [Flask](https://flask.palletsprojects.com/) + [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
- [Remix Icon](https://remixicon.com/)
- APIs: OpenAI, Deepgram, Gladia
