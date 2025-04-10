import os
import time
import threading
import torch
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename
import openai
import whisper
import requests
import json
import re # Make sure re is imported

app = Flask(__name__, template_folder=".")
app.config['SECRET_KEY'] = 'secret!'
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# Configure SocketIO with longer ping timeout to prevent disconnections
socketio = SocketIO(app, async_mode='threading', ping_timeout=60, ping_interval=25, cors_allowed_origins="*")

# Global dictionaries to track cancellations and saved transcriptions
# Structure: { job_id: { model_option: { ..., elapsed_time: int } } }
transcriptions = {}
cancellations = {}

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

# Each entry is wrapped as a transcript-segment to allow real-time highlighting when media is playing
def generate_synced_transcript_from_segments(segments):
    transcript_output = ""
    for seg in segments:
        transcript_output += f'<div class="transcript-segment" data-start="{seg["start"]}" data-end="{seg["end"]}">'
        transcript_output += f'<button class="seek-btn btn btn-ghost btn-icon btn-sm" title="Seek to here"><i class="ri-play-circle-line"></i></button>'
        transcript_output += seg["text"].strip()
        transcript_output += "</div>\n"
    return transcript_output.strip()

def generate_numbered_transcript_from_segments(segments):
    numbered_output = ""
    for i, seg in enumerate(segments, start=1):
        numbered_output += f'<div class="transcript-segment" data-start="{seg["start"]}" data-end="{seg["end"]}">'
        numbered_output += f'<button class="seek-btn btn btn-ghost btn-icon btn-sm" title="Seek to here"><i class="ri-play-circle-line"></i></button>'
        numbered_output += f'<div class="segment-number" style="opacity: 0.5;">{i}</div>'
        numbered_output += f'<div class="segment-text">{seg["text"].strip()}</div>'
        numbered_output += "</div>\n"
    return numbered_output.strip()

def generate_srt_from_segments(segments):
    output = ""
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f'<div class="transcript-segment" data-start="{seg["start"]}" data-end="{seg["end"]}">'
        output += f'<button class="seek-btn btn btn-ghost btn-icon btn-sm" title="Seek to here"><i class="ri-play-circle-line"></i></button>'
        output += f'<div class="srt-index srt-opacity" style="opacity: 0.5;">{i}</div>'
        output += f'<div class="srt-timing srt-opacity" style="opacity: 0.5;">{start_time} --> {end_time}</div>'
        output += f'<div class="srt-text">{seg["text"].strip()}</div>'
        output += "</div>\n"
    return output.strip()

def generate_vtt_from_segments(segments):
    output = '<div class="vtt-header">WEBVTT</div>\n'
    for seg in segments:
        start_time = format_time(seg["start"]).replace(",", ".")
        end_time = format_time(seg["end"]).replace(",", ".")
        output += f'<div class="transcript-segment vtt-entry" data-start="{seg["start"]}" data-end="{seg["end"]}">'
        output += f'<button class="seek-btn btn btn-ghost btn-icon btn-sm" title="Seek to here"><i class="ri-play-circle-line"></i></button>'
        output += f'<div class="vtt-timing vtt-opacity" style="opacity: 0.5;">{start_time} --> {end_time}</div>'
        output += f'<div class="vtt-text">{seg["text"].strip()}</div>'
        output += '</div>\n'
    return output.strip()

def generate_tsv_from_segments(segments):
    # HTML table version. Each row is a transcript-segment with data attributes
    output = '<table class="tsv-table"><thead><tr><th style="opacity: 0.5;">Segment</th><th style="opacity: 0.5;">Start</th><th style="opacity: 0.5;">End</th><th>Text</th></tr></thead><tbody>'
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f'<tr class="transcript-segment" data-start="{seg["start"]}" data-end="{seg["end"]}">'
        output += f'<td style="opacity: 0.5;">{i}</td><td style="opacity: 0.5;">{start_time}</td><td style="opacity: 0.5;">{end_time}</td>'
        output += f'<td>{seg["text"].strip()}</td></tr>'
    output += '</tbody></table>'
    return output

def generate_plain_tsv_from_segments(segments):
    # Plain text TSV (for download)
    output = "Segment\tStart\tEnd\tText\n"
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f"{i}\t{start_time}\t{end_time}\t{seg['text'].strip()}\n"
    return output.strip()

def generate_plain_srt_from_segments(segments):
    """Generate a plain text SRT file format from segments for download"""
    output = ""
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f"{i}\n{start_time} --> {end_time}\n{seg['text'].strip()}\n\n"
    return output.strip()

def generate_plain_srt_from_deepgram(result):
    """Generate SRT format directly from Deepgram API response structure"""
    output = ""
    index = 1
    segments_for_fallback = [] # Keep track for fallback

    # Check for utterances first (preferred with Nova-3)
    if "results" in result and "utterances" in result["results"]:
        utterances = result["results"]["utterances"]
        print(f"Generating SRT from {len(utterances)} utterances")
        for i, utterance in enumerate(utterances, start=1):
            start_time = format_time(utterance["start"])
            end_time = format_time(utterance["end"])
            text = utterance["transcript"].strip()
            output += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            segments_for_fallback.append({"start": utterance["start"], "end": utterance["end"], "text": text}) # Collect for fallback
        return output.strip()

    # If no utterances, try alternatives path
    elif "results" in result and "channels" in result["results"] and len(result["results"]["channels"]) > 0:
        channels = result["results"]["channels"]
        for channel_idx, channel in enumerate(channels):
            if "alternatives" in channel and len(channel["alternatives"]) > 0:
                alternative = channel["alternatives"][0]
                # Try paragraphs first
                if "paragraphs" in alternative:
                    paragraphs_data = alternative["paragraphs"]
                    paragraphs = paragraphs_data.get("paragraphs", paragraphs_data) # Handle both dict and list
                    print(f"Generating SRT from {len(paragraphs)} paragraphs")
                    for para in paragraphs:
                        start = para.get("start", 0)
                        end = para.get("end", 0)
                        text = None
                        if "sentences" in para and para["sentences"]:
                            for sentence in para["sentences"]:
                                s_start = sentence.get("start", start)
                                s_end = sentence.get("end", end)
                                s_text = sentence.get("text", "").strip()
                                if s_text:
                                    output += f"{index}\n{format_time(s_start)} --> {format_time(s_end)}\n{s_text}\n\n"
                                    segments_for_fallback.append({"start": s_start, "end": s_end, "text": s_text})
                                    index += 1
                            continue
                        elif "transcript" in para:
                            text = para["transcript"]
                        if text:
                            text = text.strip()
                            output += f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"
                            segments_for_fallback.append({"start": start, "end": end, "text": text})
                            index += 1
                    if output: return output.strip() # Return if paragraphs worked

                # If no paragraphs, try words
                elif "words" in alternative:
                    words = alternative["words"]
                    current_text = ""
                    current_start = 0
                    current_end = 0 # Initialize end time
                    print(f"Generating SRT from {len(words)} words")
                    for word in words:
                        word_text = word.get("punctuated_word", word.get("word", "")).strip()
                        if not word_text: continue # Skip empty words

                        if not current_text:
                            current_start = word["start"]

                        current_text += (" " if current_text else "") + word_text
                        current_end = word["end"] # Update end time with each word

                        # Create a new segment after punctuation or every ~12 words
                        if (word_text.endswith((".", "?", "!")) or len(current_text.split()) >= 12):
                            output += f"{index}\n{format_time(current_start)} --> {format_time(current_end)}\n{current_text.strip()}\n\n"
                            segments_for_fallback.append({"start": current_start, "end": current_end, "text": current_text.strip()})
                            index += 1
                            current_text = ""

                    # Add the last segment if any
                    if current_text:
                        output += f"{index}\n{format_time(current_start)} --> {format_time(current_end)}\n{current_text.strip()}\n\n"
                        segments_for_fallback.append({"start": current_start, "end": current_end, "text": current_text.strip()})
                    if output: return output.strip() # Return if words worked

    # Fallback if no segments could be parsed
    print("Falling back to segment-based SRT generation from Deepgram response.")
    if not segments_for_fallback: # If we didn't collect any segments during processing
        segments_for_fallback = extract_segments_from_deepgram_response(result) # Extract fresh
    return generate_plain_srt_from_segments(segments_for_fallback)


def calculate_token_count(text, model_name="gpt-4"):
    """Estimates token count using a simple word-based approximation."""
    if not text:
        return 0
    
    return len(text.split())

@app.route('/', methods=['GET'])
def index():
    return render_template('Kaira_Transcribe_Panel.html')

# ------------------------------
# API Transcription Routes
# ------------------------------
@app.route('/transcribe', methods=['POST'])
def transcribe_api():
    api_key = request.form.get('api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')

    if not api_key or not file or not job_id:
        return jsonify({"error": "API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(api_transcription, job_id, filepath, api_key)

    return jsonify({"job_id": job_id, "status": "started", "model_option": "whisper-1"})

@app.route('/transcribe_deepgram', methods=['POST'])
def transcribe_deepgram():
    api_key = request.form.get('deepgram_api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')
    summarize_enabled = request.form.get('summarize_enabled') == 'true'
    topics_enabled = request.form.get('topics_enabled') == 'true'

    if not api_key or not file or not job_id:
        return jsonify({"error": "Deepgram API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(deepgram_transcription, job_id, filepath, api_key, summarize_enabled, topics_enabled)

    return jsonify({"job_id": job_id, "status": "started", "model_option": "nova-3"})

@app.route('/transcribe_gladia', methods=['POST'])
def transcribe_gladia():
    api_key = request.form.get('gladia_api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')

    if not api_key or not file or not job_id:
        return jsonify({"error": "Gladia API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(gladia_transcription, job_id, filepath, api_key)

    return jsonify({"job_id": job_id, "status": "started", "model_option": "gladia"})

def api_transcription(job_id, filepath, api_key):
    start_time = time.time() # Start timer
    try:
        if cancellations.get(job_id, False): return
        print(f"Starting API transcription for job ID: {job_id}")
        socketio.emit("progress_update", { "job_id": job_id, "progress": 30, "message": "Starting OpenAI API transcription", "elapsed": 0, "remaining": 0 }, room=job_id)

        openai.api_key = api_key
        with open(filepath, "rb") as audio_file:
            try:
                client = openai.OpenAI(api_key=api_key)
                response = client.audio.transcriptions.create( model="whisper-1", file=audio_file, response_format="verbose_json", timestamp_granularities=["segment"] )
                if not isinstance(response, dict): response = response.model_dump()
            except (AttributeError, ImportError, TypeError):
                 response = openai.Audio.transcribe( "whisper-1", audio_file, response_format="verbose_json" )
                 if not isinstance(response, dict): response = response.__dict__

        if cancellations.get(job_id, False): return

        segments = []
        if isinstance(response, dict) and "segments" in response:
            socketio.emit("progress_update", { "job_id": job_id, "progress": 70, "message": "Processing transcript segments", "elapsed": 0, "remaining": 0 }, room=job_id)
            segments = response["segments"]
        elif isinstance(response, dict) and "text" in response:
             text = response["text"]
             words = text.split(); chunk_size = 20
             for i in range(0, len(words), chunk_size):
                 chunk = " ".join(words[i:i+chunk_size]); start_time_est = i / 2.5; end_time_est = min(len(words), i + chunk_size) / 2.5
                 segments.append({"start": start_time_est, "end": end_time_est, "text": chunk})

        if cancellations.get(job_id, False): return

        srt_text = generate_srt_from_segments(segments)
        default_transcript = generate_synced_transcript_from_segments(segments)
        numbered_transcript = generate_numbered_transcript_from_segments(segments)
        vtt_text = generate_vtt_from_segments(segments)
        tsv_html = generate_tsv_from_segments(segments)
        tsv_plain = generate_plain_tsv_from_segments(segments)
        plain_srt_for_download = generate_plain_srt_from_segments(segments)
        plain_text_for_tokens = html_to_plain_text(default_transcript)
        token_count = calculate_token_count(plain_text_for_tokens, "whisper-1")
        detected_language = response.get("language", "auto-detected") # Get detected language
        elapsed = time.time() - start_time # Calculate elapsed time

        if job_id not in transcriptions: transcriptions[job_id] = {}
        transcriptions[job_id]["whisper-1"] = {
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "task": "transcribe", "language": detected_language, "token_count": token_count, "elapsed_time": int(elapsed) # Store elapsed time
        }
        socketio.emit("progress_update", { "job_id": job_id, "progress": 100, "message": "Transcription complete", "elapsed": 0, "remaining": 0 }, room=job_id)
        socketio.emit("transcription_complete", {
            "job_id": job_id, "model_option": "whisper-1", "task": "transcribe", "language": detected_language,
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "token_count": token_count, "elapsed_time": int(elapsed) # Send elapsed time
        }, room=job_id)
    except Exception as e:
        print(f"Error in API transcription: {str(e)}")
        if not cancellations.get(job_id, False): socketio.emit("transcription_failed", { "job_id": job_id, "error": str(e) }, room=job_id)
    finally:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except: pass
        cancellations.pop(job_id, None)


def extract_segments_from_deepgram_response(result):
    segments = []
    try:
        if not result or "results" not in result: return []
        results_data = result["results"]
        if "utterances" in results_data and results_data["utterances"]:
            print(f"Extracting segments from {len(results_data['utterances'])} utterances.")
            for utterance in results_data["utterances"]:
                segments.append({"start": utterance.get("start", 0),"end": utterance.get("end", 0),"text": utterance.get("transcript", "").strip()})
            if segments: return segments
        if "channels" in results_data and results_data["channels"]:
            alternative = results_data["channels"][0].get("alternatives", [{}])[0]
            if "paragraphs" in alternative and alternative["paragraphs"]:
                paragraphs_data = alternative["paragraphs"]
                paragraphs_list = paragraphs_data.get("paragraphs", paragraphs_data) # Handle dict or list
                print(f"Extracting segments from {len(paragraphs_list)} paragraphs.")
                for para in paragraphs_list:
                    para_start = para.get("start", 0); para_end = para.get("end", 0)
                    if "sentences" in para and para["sentences"]:
                        for sentence in para["sentences"]: segments.append({"start": sentence.get("start", para_start),"end": sentence.get("end", para_end),"text": sentence.get("text", "").strip()})
                    elif "transcript" in para and para["transcript"].strip(): segments.append({"start": para_start,"end": para_end,"text": para["transcript"].strip()})
                if segments: return segments
            if "sentences" in alternative and alternative["sentences"]:
                 print(f"Extracting segments from {len(alternative['sentences'])} sentences.")
                 for sentence in alternative["sentences"]: segments.append({"start": sentence.get("start", 0),"end": sentence.get("end", 0),"text": sentence.get("text", "").strip()})
                 if segments: return segments
            if "words" in alternative and alternative["words"]:
                words = alternative["words"]; print(f"Extracting segments from {len(words)} words.")
                current_segment = {"text": "", "start": 0, "end": 0}; words_in_segment = 0
                for word in words:
                    if not current_segment["text"]: current_segment["start"] = word.get("start", 0)
                    current_segment["text"] += (" " if current_segment["text"] else "") + word.get("punctuated_word", word.get("word", ""))
                    current_segment["end"] = word.get("end", 0); words_in_segment += 1
                    ends_with_punctuation = word.get("punctuated_word", "").strip().endswith(('.', '?', '!'))
                    if (ends_with_punctuation or words_in_segment >= 15) and current_segment["text"].strip():
                        segments.append(dict(current_segment))
                        current_segment = {"text": "", "start": word.get("end", 0), "end": word.get("end", 0)}; words_in_segment = 0
                if current_segment["text"].strip(): segments.append(current_segment)
                if segments: return segments
            if "transcript" in alternative and alternative["transcript"]:
                full_transcript = alternative["transcript"].strip(); print(f"Warning: Falling back to splitting full transcript.")
                sentences = re.split(r'(?<=[.!?])\s+', full_transcript)
                duration = result.get("metadata", {}).get("duration", len(sentences) * 3); start_time = 0
                time_per_sentence = duration / len(sentences) if sentences else 0
                for sentence in sentences:
                    sentence_text = sentence.strip()
                    if sentence_text:
                        end_time = start_time + time_per_sentence
                        segments.append({"start": start_time,"end": end_time,"text": sentence_text})
                        start_time = end_time
                if segments: return segments
    except Exception as e: print(f"Error during Deepgram segment extraction: {str(e)}")
    if not segments: print("Warning: Could not extract any segments from Deepgram response.")
    return segments

def deepgram_transcription(job_id, filepath, api_key, summarize_enabled=False, topics_enabled=False):
    start_time = time.time() # Start timer
    summary_text = None
    topics_data = None
    detected_language = "multi" # Supports 10 languages (multi: English, Spanish, French, German, Hindi, Russian, Portuguese, Japanese, Italian and Dutch)

    try:
        if cancellations.get(job_id, False): return
        print(f"Starting Deepgram transcription for job ID: {job_id}")
        socketio.emit("progress_update", { "job_id": job_id, "progress": 30, "message": f"Starting Deepgram Nova-3 transcription", "elapsed": 0, "remaining": 0 }, room=job_id)

        file_ext = os.path.splitext(filepath)[1].lower()
        mime_types = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/m4a", ".mp4": "video/mp4", ".webm": "video/webm", ".ogg": "audio/ogg", ".flac": "audio/flac" }
        content_type = mime_types.get(file_ext, "audio/mpeg")
        print(f"Using content type {content_type} for file {os.path.basename(filepath)}")

        headers = { "Authorization": f"Token {api_key}", "Content-Type": content_type }
        with open(filepath, "rb") as file: file_content = file.read()

        # Build parameters
        params = {
            "model": "nova-3", "language": "multi", "diarize": "true", "punctuate": "true",
            "paragraphs": "true", "utterances": "true", "smart_format": "true"
        }

        socketio.emit("progress_update", { "job_id": job_id, "progress": 50, "message": "Processing with Deepgram API", "elapsed": 0, "remaining": 0 }, room=job_id)
        print(f"Sending request to Deepgram API with params: {params}")
        response = requests.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, data=file_content)

        if response.status_code != 200:
            error_message = f"Deepgram API error: {response.status_code}, {response.text}"
            print(error_message); raise Exception(error_message)

        if cancellations.get(job_id, False): return

        try:
            result = response.json()
            print("Deepgram API response received.")
            # Check metadata for detected language
            detected_language = result.get("metadata", {}).get("language", "multi")
        except json.JSONDecodeError as e: raise Exception(f"Invalid JSON response from Deepgram: {str(e)}")

        socketio.emit("progress_update", { "job_id": job_id, "progress": 70, "message": "Processing transcript segments", "elapsed": 0, "remaining": 0 }, room=job_id)
        segments = extract_segments_from_deepgram_response(result)
        full_transcript_text = ""
        try: # Extract full transcript for AI features
            if "results" in result and "channels" in result["results"] and result["results"]["channels"]:
                alternatives = result["results"]["channels"][0].get("alternatives", [{}])
                if alternatives and "transcript" in alternatives[0]: full_transcript_text = alternatives[0]["transcript"]
        except Exception as text_extract_err: print(f"Warning: Could not extract full transcript for AI features: {text_extract_err}")

        # AI Feature Calls (Summarize/Topics)
        if full_transcript_text and (summarize_enabled or topics_enabled):
            print(f"Checking for AI features...")
            text_intel_headers = { "Authorization": f"Token {api_key}", "Content-Type": "application/json" }
            text_intel_payload = { "text": full_transcript_text }
            # Note: Deepgram's AI features (summarization and topic detection) only work with English content
            text_intel_params = { "language": "en" }  # Must use English for these features

            if summarize_enabled:
                socketio.emit("progress_update", { "job_id": job_id, "progress": 75, "message": "Requesting summary...", "elapsed": 0, "remaining": 0 }, room=job_id)
                print("Requesting Summarization...")
                try:
                    summary_params = text_intel_params.copy(); summary_params["summarize"] = "true"
                    summary_response = requests.post("https://api.deepgram.com/v1/read", params=summary_params, headers=text_intel_headers, json=text_intel_payload)
                    if summary_response.status_code == 200:
                        summary_result = summary_response.json()
                        if "results" in summary_result and "summary" in summary_result["results"]:
                            summary_text = summary_result["results"]["summary"].get("text")
                            print(f"Summary received.")
                        else: print(f"Warning: Summarization response structure unexpected.")
                    else: print(f"Warning: Summarization API error {summary_response.status_code}: {summary_response.text}")
                except Exception as summary_err: print(f"Error during Summarization API call: {summary_err}")

            if topics_enabled:
                socketio.emit("progress_update", { "job_id": job_id, "progress": 80, "message": "Requesting topic detection...", "elapsed": 0, "remaining": 0 }, room=job_id)
                print("Requesting Topic Detection...")
                try:
                    topics_params = text_intel_params.copy(); topics_params["topics"] = "true"
                    topics_response = requests.post("https://api.deepgram.com/v1/read", params=topics_params, headers=text_intel_headers, json=text_intel_payload)
                    if topics_response.status_code == 200:
                        topics_result = topics_response.json()
                        if "results" in topics_result and "topics" in topics_result["results"]:
                            topics_data = topics_result["results"]["topics"]
                            print(f"Topic data received.")
                        else: print(f"Warning: Topic Detection response structure unexpected.")
                    else: print(f"Warning: Topic Detection API error {topics_response.status_code}: {topics_response.text}")
                except Exception as topics_err: print(f"Error during Topic Detection API call: {topics_err}")

        # Segment Fallback Logic
        if not segments:
            print("Critical: Segment extraction failed. Attempting fallback.")
            if full_transcript_text:
                 words = full_transcript_text.split(); chunk_size = 15
                 duration = result.get("metadata", {}).get("duration", len(words) * 0.5)
                 time_per_word = duration / len(words) if words else 0
                 current_word_index = 0
                 for i in range(0, len(words), chunk_size):
                     chunk = " ".join(words[i:i+chunk_size])
                     start_time_est = current_word_index * time_per_word
                     end_word_index = min(len(words), i + chunk_size)
                     end_time_est = end_word_index * time_per_word
                     segments.append({ "start": start_time_est, "end": end_time_est, "text": chunk })
                     current_word_index = end_word_index
            else: segments = [{"start": 0, "end": 1, "text": "Error: Could not process Deepgram response."}]


        if cancellations.get(job_id, False): return

        print(f"Using {len(segments)} segments for formatting.")

        srt_text = generate_srt_from_segments(segments)
        default_transcript = generate_synced_transcript_from_segments(segments)
        numbered_transcript = generate_numbered_transcript_from_segments(segments)
        vtt_text = generate_vtt_from_segments(segments)
        tsv_html = generate_tsv_from_segments(segments)
        tsv_plain = generate_plain_tsv_from_segments(segments)
        plain_srt_for_download = generate_plain_srt_from_deepgram(result)
        plain_text_for_tokens = html_to_plain_text(default_transcript)
        token_count = calculate_token_count(plain_text_for_tokens, "nova-3")
        elapsed = time.time() - start_time # Calculate elapsed time

        if job_id not in transcriptions: transcriptions[job_id] = {}
        transcriptions[job_id]["nova-3"] = {
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "task": "transcribe", "language": detected_language, "token_count": token_count,
            "summary_text": summary_text, "topics_data": topics_data, "elapsed_time": int(elapsed) # Store elapsed time
        }

        socketio.emit("progress_update", { "job_id": job_id, "progress": 100, "message": "Transcription complete", "elapsed": 0, "remaining": 0 }, room=job_id)
        print(f"Sending transcription_complete event for job {job_id}")
        socketio.emit("transcription_complete", {
            "job_id": job_id, "model_option": "nova-3", "task": "transcribe",
            "language": detected_language,
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "token_count": token_count, "summary_text": summary_text, "topics_data": topics_data,
            "elapsed_time": int(elapsed) # Send elapsed time
        }, room=job_id)
        print(f"Sent transcription_complete event for job {job_id}")

    except Exception as e:
        print(f"Error in Deepgram transcription: {str(e)}")
        if not cancellations.get(job_id, False):
            print(f"Sending transcription_failed event for job {job_id}")
            socketio.emit("transcription_failed", { "job_id": job_id, "error": str(e) }, room=job_id)
    finally:
        if os.path.exists(filepath):
            try: os.remove(filepath); print(f"Removed temporary file: {filepath}")
            except: print(f"Failed to remove temporary file: {filepath}")
        cancellations.pop(job_id, None)
        print(f"Transcription job {job_id} completed")

def gladia_transcription(job_id, filepath, api_key):
    start_time = time.time() # Start timer
    detected_language = "en" # Default to English as we force it
    try:
        if cancellations.get(job_id, False): return
        print(f"Starting Gladia transcription (forced English) for job ID: {job_id}")
        socketio.emit("progress_update", { "job_id": job_id, "progress": 30, "message": f"Starting Gladia transcription", "elapsed": 0, "remaining": 0 }, room=job_id)

        # Step 1: Upload
        with open(filepath, "rb") as audio_file:
            filename = os.path.basename(filepath)
            upload_headers = { "x-gladia-key": api_key }
            file_ext = os.path.splitext(filename)[1].lower()
            mime_types = { ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/m4a", ".mp4": "video/mp4", ".webm": "video/webm", ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac", ".opus": "audio/opus" }
            mime_type = mime_types.get(file_ext, "audio/mpeg")
            files = { "audio": (filename, audio_file, mime_type) }
            socketio.emit("progress_update", { "job_id": job_id, "progress": 40, "message": "Uploading file to Gladia", "elapsed": 0, "remaining": 0 }, room=job_id)
            print(f"Uploading file to Gladia: {filename} with type {mime_type}")
            try:
                upload_response = requests.post( "https://api.gladia.io/v2/upload", headers=upload_headers, files=files )
                if upload_response.status_code != 200: raise Exception(f"Gladia API upload error: {upload_response.status_code}, {upload_response.text}")
                upload_result = upload_response.json(); audio_url = upload_result.get("audio_url")
                if not audio_url: raise Exception("Failed to get audio URL from Gladia upload response")
                print(f"Successfully uploaded file to Gladia, received URL: {audio_url}")
            except Exception as e: raise Exception(f"Error during file upload to Gladia: {str(e)}")

        if cancellations.get(job_id, False): return

        # Step 2: Request transcription - force language 'en'
        socketio.emit("progress_update", { "job_id": job_id, "progress": 50, "message": "Processing with Gladia API (English)", "elapsed": 0, "remaining": 0 }, room=job_id)
        transcription_headers = { "x-gladia-key": api_key, "Content-Type": "application/json" }
        transcription_data = {
            "audio_url": audio_url, "language": "en", "detect_language": False, "translation": False,
            "enable_code_switching": False, "diarization": True,
            "diarization_config": { "number_of_speakers": 2, "min_speakers": 1, "max_speakers": 5, "enhanced": True },
            "subtitles": True, "subtitles_config": { "formats": ["srt", "vtt"], "maximum_characters_per_row": 40, "maximum_rows_per_caption": 2 }
        }
        print(f"Gladia API request (transcription only, forced English):")
        print(f"  - language: {transcription_data.get('language')}, detect_language: {transcription_data.get('detect_language')}")
        try:
            transcription_response = requests.post( "https://api.gladia.io/v2/pre-recorded", headers=transcription_headers, json=transcription_data )
            if transcription_response.status_code not in [200, 201]: raise Exception(f"Gladia API transcription error: {transcription_response.status_code}, {transcription_response.text}")
            transcription_result = transcription_response.json()
            transcription_id = transcription_result.get("id"); result_url = transcription_result.get("result_url")
            if not transcription_id or not result_url: raise Exception("Failed to get transcription ID or result URL from Gladia response")
            print(f"Successfully submitted transcription request to Gladia, ID: {transcription_id}")
        except Exception as e: raise Exception(f"Error during transcription request to Gladia: {str(e)}")

        if cancellations.get(job_id, False): return

        # Step 3: Poll for results
        socketio.emit("progress_update", { "job_id": job_id, "progress": 70, "message": "Waiting for transcription results", "elapsed": 0, "remaining": 0 }, room=job_id)
        result = None; max_retries = 60; retry_count = 0
        while retry_count < max_retries:
            if cancellations.get(job_id, False): return
            try:
                result_response = requests.get( result_url, headers={"x-gladia-key": api_key} )
                if result_response.status_code == 200:
                    result = result_response.json(); status = result.get("status")
                    if status == "done": print("Transcription completed successfully"); break
                    elif status == "error": raise Exception(f"Gladia transcription error: {result.get('error', 'Unknown error')}")
                    progress = min(90, 70 + int((retry_count / max_retries) * 20))
                    socketio.emit("progress_update", { "job_id": job_id, "progress": progress, "message": "Processing transcription...", "elapsed": 0, "remaining": 0 }, room=job_id)
                elif result_response.status_code == 404: print("Result not ready yet, waiting...")
                else: print(f"Unexpected response from Gladia result URL: {result_response.status_code}")
            except Exception as e: print(f"Error while polling for results: {str(e)}")
            retry_count += 1; time.sleep(10)
        if not result or result.get("status") != "done": raise Exception("Transcription timed out or failed")

        # Step 4: Process the results
        socketio.emit("progress_update", { "job_id": job_id, "progress": 90, "message": "Processing transcript segments", "elapsed": 0, "remaining": 0 }, room=job_id)
        transcript_data = result.get("result", {}); print("Gladia API result structure received.")
        segments = []
        detected_language = transcript_data.get("language", "en") # Get language if provided, default 'en'

        # Extract segments (same logic as before)
        if "transcription" in transcript_data and isinstance(transcript_data["transcription"], dict) and "utterances" in transcript_data["transcription"]:
            utterances = transcript_data["transcription"]["utterances"]; print(f"Found {len(utterances)} utterances")
            for u in utterances: segments.append({ "start": u.get("start", 0), "end": u.get("end", 0), "text": str(u.get("text", "")).strip() })
        elif "transcription" in transcript_data and isinstance(transcript_data["transcription"], dict) and "full_transcript" in transcript_data["transcription"]:
             full_text = transcript_data["transcription"]["full_transcript"]; print(f"Using full_transcript fallback...")
             lines = full_text.strip().split('\n'); duration = transcript_data.get("metadata", {}).get("duration", 10)
             time_per_line = duration / len(lines) if lines else 0; start_t = 0
             for line in lines:
                 if line.strip(): end_t = start_t + time_per_line; segments.append({ "start": start_t, "end": end_t, "text": line.strip() }); start_t = end_t
        elif "subtitles" in transcript_data and transcript_data.get("subtitles", {}).get("srt"):
            srt_content = transcript_data["subtitles"]["srt"]; print("Using SRT subtitle fallback.")
            srt_blocks = re.split(r'\n\n+', srt_content.strip())
            for block in srt_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    try:
                        time_line = lines[1]; parts = time_line.split(' --> ')
                        if len(parts) == 2: s_time = parse_srt_time(parts[0]); e_time = parse_srt_time(parts[1]); txt = ' '.join(lines[2:])
                        segments.append({ "start": s_time, "end": e_time, "text": txt.strip() })
                    except Exception as srt_err: print(f"Warning: Could not parse SRT block: {srt_err}")
        if not segments:
            print("WARNING: No segments extracted. Creating dummy segment."); segments.append({ "start": 0, "end": 1, "text": "Transcription completed, but no text segments extracted."})
        print(f"Generated {len(segments)} segments.")

        srt_text = generate_srt_from_segments(segments)
        default_transcript = generate_synced_transcript_from_segments(segments)
        numbered_transcript = generate_numbered_transcript_from_segments(segments)
        vtt_text = generate_vtt_from_segments(segments)
        tsv_html = generate_tsv_from_segments(segments)
        tsv_plain = generate_plain_tsv_from_segments(segments)
        plain_srt_for_download = transcript_data.get("subtitles", {}).get("srt", generate_plain_srt_from_segments(segments))
        plain_text_for_tokens = html_to_plain_text(default_transcript)
        token_count = calculate_token_count(plain_text_for_tokens)
        elapsed = time.time() - start_time # Calculate elapsed time

        if job_id not in transcriptions: transcriptions[job_id] = {}
        transcriptions[job_id]["gladia"] = {
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "task": "transcribe", "language": detected_language, "token_count": token_count, "elapsed_time": int(elapsed) # Store elapsed
        }

        socketio.emit("progress_update", { "job_id": job_id, "progress": 100, "message": "Transcription complete", "elapsed": 0, "remaining": 0 }, room=job_id)
        socketio.emit("transcription_complete", {
            "job_id": job_id, "model_option": "gladia", "task": "transcribe", "language": detected_language,
            "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
            "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
            "token_count": token_count, "elapsed_time": int(elapsed) # Send elapsed
        }, room=job_id)

    except Exception as e:
        print(f"Error in Gladia transcription: {str(e)}")
        if not cancellations.get(job_id, False): socketio.emit("transcription_failed", { "job_id": job_id, "error": str(e) }, room=job_id)
    finally:
        if os.path.exists(filepath):
            try: os.remove(filepath); print(f"Removed temporary file: {filepath}")
            except: print(f"Failed to remove temporary file: {filepath}")
        cancellations.pop(job_id, None)
        print(f"Transcription job {job_id} completed")


def parse_srt_time(time_str):
    # (Function remains the same)
    try:
        time_str_cleaned = time_str.replace(',', '.')
        parts = time_str_cleaned.split(':')
        if len(parts) == 3:
            hours = int(parts[0]); minutes = int(parts[1]); seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else: print(f"Warning: Could not parse SRT time format: {time_str}"); return 0
    except ValueError: print(f"Warning: ValueError parsing SRT time: {time_str}"); return 0
    except Exception as e: print(f"Warning: Unexpected error parsing SRT time '{time_str}': {e}"); return 0

@socketio.on("cancel_transcription")
def handle_cancel_transcription(data):
    job_id = data.get("job_id")
    if job_id:
        print(f"Cancellation requested for job: {job_id}")
        cancellations[job_id] = True
        emit("transcription_cancelled", {"job_id": job_id}, room=job_id)

@socketio.on("join")
def on_join(data):
    job_id = data.get("job_id")
    if job_id:
        join_room(job_id)
        print(f"Socket client joined room: {job_id}")
        # emit("joined", {"job_id": job_id}) # Optional: Confirm join

@app.route('/download_srt', methods=['POST'])
def download_srt():
    # (This function remains the same)
    output_type = request.form.get('output_type', 'srt')
    job_id = request.form.get('job_id')
    model = request.form.get('model', '')
    if output_type == "srt": ext = "srt"
    elif output_type == "vtt": ext = "vtt"
    elif output_type == "tsv": ext = "tsv"
    else: ext = "txt"
    output_text = ""
    if 'srt_text' in request.form:
        output_text = request.form.get('srt_text', '')
    elif job_id and job_id in transcriptions:
        model_results = None
        for model_name, results in transcriptions[job_id].items():
            if model and model == model_name: model_results = results; break
            if not model: model_results = results; break
        if model_results:
            if output_type == "srt" and "plain_srt" in model_results: output_text = model_results["plain_srt"]; print("Using plain SRT format for download")
            elif output_type == "srt": output_text = html_to_plain_srt(model_results.get("srt_text", ""))
            elif output_type == "vtt": output_text = html_to_plain_vtt(model_results.get("vtt_text", ""))
            elif output_type == "tsv" and "tsv_plain" in model_results: output_text = model_results["tsv_plain"]
            else: output_text = html_to_plain_text(model_results.get("default_transcript", ""))
    if not output_text: return jsonify({"error": "No transcript data found"}), 400
    output_file = os.path.join(app.config['TEMP_FOLDER'], f"transcription.{ext}")
    with open(output_file, "w", encoding="utf-8") as f: f.write(output_text)
    return send_file(output_file, as_attachment=True, download_name=f"transcription.{ext}")

def html_to_plain_text(html_content):
    # (Function remains the same)
    if not html_content: return ""
    import re
    plain_text = re.sub(r'<[^>]*>', ' ', html_content)
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()
    return plain_text

def html_to_plain_srt(srt_html):
    # (Function remains the same)
    if not srt_html: return ""
    import re; output = ""
    segments = re.findall(r'<div class="transcript-segment"[^>]*>.*?<div class="srt-index[^>]*>(\d+)</div>.*?<div class="srt-timing[^>]*>([^<]*)</div>.*?<div class="srt-text[^>]*>(.*?)</div>.*?</div>', srt_html, re.DOTALL)
    for index, timing, text in segments:
        clean_text = re.sub(r'<[^>]*>', '', text).strip(); clean_timing = timing.strip()
        output += f"{index}\n{clean_timing}\n{clean_text}\n\n"
    return output.strip()

def html_to_plain_vtt(vtt_html):
    # (Function remains the same)
    if not vtt_html: return ""
    import re; output = ""
    header_match = re.search(r'<div class="vtt-header">(.*?)</div>', vtt_html, re.IGNORECASE)
    output += (header_match.group(1).strip() + "\n\n") if header_match else "WEBVTT\n\n"
    segments = re.findall(r'<div class="transcript-segment vtt-entry"[^>]*>.*?<div class="vtt-timing[^>]*>([^<]*)</div>.*?<div class="vtt-text[^>]*>(.*?)</div>.*?</div>', vtt_html, re.DOTALL)
    for timing, text in segments:
        clean_text = re.sub(r'<[^>]*>', '', text).strip(); clean_timing = timing.strip()
        output += f"{clean_timing}\n{clean_text}\n\n"
    return output.strip()

# ------------------------------
# Local Transcription Route
# ------------------------------
@app.route('/transcribe_local', methods=['POST'])
def transcribe_local():
    file = request.files.get('file')
    model_option = request.form.get('model_option', 'tiny') # Extract model_option
    job_id = request.form.get('job_id')

    print(f"Received local request: model={model_option}, job={job_id}")

    if not file or not job_id or not model_option:
        return jsonify({"error": "File, job_id, and model_option are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(fast_transcription, job_id, filepath, model_option)
    return jsonify({"job_id": job_id, "status": "started", "model_option": model_option})

def fast_transcription(job_id, file_path, model_option):
    start_time = time.time()
    detected_language = "auto-detected" # Initialize
    try:
        if cancellations.get(job_id, False): return
        print(f"Loading local model: {model_option} for transcription")
        socketio.emit("progress_update", { "job_id": job_id, "progress": 10, "message": f"Loading {model_option} model", "elapsed": 0, "remaining": 0 }, room=job_id)
        try:
            model = whisper.load_model(model_option) # Use the passed model_option
            print(f"Model loaded: {model_option}")
        except ValueError as e:
            print(f"Error loading model {model_option}: {str(e)}")
            if "is not a valid model name" in str(e): socketio.emit("transcription_failed", { "job_id": job_id, "error": f"Invalid model name: {model_option}. Ensure it's installed." }, room=job_id)
            else: socketio.emit("transcription_failed", { "job_id": job_id, "error": str(e) }, room=job_id)
            return # Stop execution if model loading fails

        if cancellations.get(job_id, False): return
        socketio.emit("progress_update", { "job_id": job_id, "progress": 30, "message": f"Transcribing audio", "elapsed": int(time.time() - start_time), "remaining": 0 }, room=job_id)
        options = { "task": "transcribe", "language": None, "fp16": torch.cuda.is_available(), "temperature": 0, "best_of": 5, "beam_size": 5, "patience": 1.0, "suppress_tokens": "-1", "word_timestamps": True, "condition_on_previous_text": True, "initial_prompt": None, }
        if model_option == "turbo": socketio.emit("progress_update", { "job_id": job_id, "progress": 35, "message": "Using Turbo model", "elapsed": int(time.time() - start_time), "remaining": 0 }, room=job_id)

        if cancellations.get(job_id, False): return
        result = model.transcribe(file_path, **options)
        if cancellations.get(job_id, False): return

        socketio.emit("progress_update", { "job_id": job_id, "progress": 70, "message": "Processing transcript", "elapsed": int(time.time() - start_time), "remaining": 0 }, room=job_id)
        segments = result.get("segments", [])
        detected_language = result.get("language", "auto-detected")

        # Segment improvement logic... (same as before)
        improved_segments = []
        for i, seg in enumerate(segments):
             text = seg["text"].strip()
             if text:
                 is_first = (i == 0)
                 prev_ends_sentence = not is_first and improved_segments[-1]["text"].strip().endswith(('.', '!', '?'))
                 if (is_first or prev_ends_sentence) and text[0].isalpha() and not text[0].isupper():
                      text = text[0].upper() + text[1:]

                 is_last = (i == len(segments) - 1)
                 next_starts_upper = False
                 if not is_last:
                     next_seg_text = segments[i+1]["text"].strip()
                     if next_seg_text and next_seg_text[0].isupper(): next_starts_upper = True

                 if not text.endswith(('.', '!', '?', ',', ':', ';', '"', "'")) and (is_last or next_starts_upper):
                      text += '.'
             improved_segments.append({"start": seg["start"], "end": seg["end"], "text": text})

        if cancellations.get(job_id, False): return

        srt_text = generate_srt_from_segments(improved_segments)
        default_transcript = generate_synced_transcript_from_segments(improved_segments)
        numbered_transcript = generate_numbered_transcript_from_segments(improved_segments)
        vtt_text = generate_vtt_from_segments(improved_segments)
        tsv_html = generate_tsv_from_segments(improved_segments)
        tsv_plain = generate_plain_tsv_from_segments(improved_segments)
        plain_srt_for_download = generate_plain_srt_from_segments(improved_segments)
        plain_text_for_tokens = html_to_plain_text(default_transcript)
        token_count = calculate_token_count(plain_text_for_tokens)
        elapsed = time.time() - start_time # Calculate elapsed time

    except Exception as e:
        if not cancellations.get(job_id, False):
            print(f"Error in local transcription: {str(e)}")
            socketio.emit("transcription_failed", {"job_id": job_id, "error": str(e)}, room=job_id)
        # Reset variables on error
        srt_text, default_transcript, numbered_transcript, vtt_text, tsv_html, tsv_plain, plain_srt_for_download = [""] * 7
        token_count = 0; detected_language = "Error"; elapsed = 0
        try:
            if 'model' in locals() and hasattr(model, 'cpu'): model.cpu() # Try to move model to CPU
            torch.cuda.empty_cache()
            if os.path.exists(file_path): os.remove(file_path)
            cancellations.pop(job_id, None)
        except Exception as cleanup_err: print(f"Cleanup error: {cleanup_err}")
        return # Important to return after handling error

    if cancellations.get(job_id, False):
        try:
            if 'model' in locals() and hasattr(model, 'cpu'): model.cpu()
            torch.cuda.empty_cache()
            if os.path.exists(file_path): os.remove(file_path)
            cancellations.pop(job_id, None)
        except Exception as cleanup_err: print(f"Cancellation cleanup error: {cleanup_err}")
        return

    if job_id not in transcriptions: transcriptions[job_id] = {}
    transcriptions[job_id][model_option] = {
        "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
        "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
        "task": "transcribe", "language": detected_language, "token_count": token_count, "elapsed_time": int(elapsed) # Store elapsed
    }
    socketio.emit("progress_update", { "job_id": job_id, "progress": 100, "message": "Transcription complete", "elapsed": int(elapsed), "remaining": 0 }, room=job_id) # Also send final elapsed here for UI update
    socketio.emit("transcription_complete", {
        "job_id": job_id, "model_option": model_option, "task": "transcribe", "language": detected_language,
        "srt_text": srt_text, "default_transcript": default_transcript, "numbered_transcript": numbered_transcript,
        "vtt_text": vtt_text, "tsv_text": tsv_html, "tsv_plain": tsv_plain, "plain_srt": plain_srt_for_download,
        "token_count": token_count, "elapsed_time": int(elapsed) # Send elapsed
    }, room=job_id)

    # Final cleanup
    try:
        if 'model' in locals() and hasattr(model, 'cpu'): model.cpu(); del model
        torch.cuda.empty_cache()
        if os.path.exists(file_path): os.remove(file_path)
        cancellations.pop(job_id, None)
    except Exception as final_cleanup_err: print(f"Final cleanup error: {final_cleanup_err}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)