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
import re
import mutagen  # For audio duration, especially for GPT-4o fallback

app = Flask(__name__, template_folder=".")
# Fail-fast on oversized uploads to save IO/CPU
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB
app.config['SECRET_KEY'] = 'secret!'
TEMP_FOLDER = 'temp'
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# Configure SocketIO with longer timeouts
socketio = SocketIO(app, async_mode='threading', ping_timeout=60, ping_interval=25, cors_allowed_origins="*")

from flask_compress import Compress
Compress(app)

# Globally reuse HTTP connections to reduce TLS/TCP overhead
http = requests.Session()

# Cap torch CPU threads to avoid oversubscription on busy hosts
try:
    if 'OMP_NUM_THREADS' not in os.environ:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    pass

# Global structures
transcriptions = {}  # { job_id: { model_option: { "srt_text":..., etc. } } }
cancellations = {}   # { job_id: bool }

######################################################
# MIME TYPE MAP (shared among providers)
######################################################
mime_types_map = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/m4a",
    ".mp4": "video/mp4", ".webm": "video/webm", ".ogg": "audio/ogg",
    ".flac": "audio/flac", ".aac": "audio/aac", ".opus": "audio/opus",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv", ".3gp": "video/3gpp", ".aiff": "audio/aiff",
    ".aif": "audio/aiff"
}

######################################################
# TIME FORMATTING + TRANSCRIPT GENERATORS
# (unchanged utility functions)
######################################################
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

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
    output = "Segment\tStart\tEnd\tText\n"
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f"{i}\t{start_time}\t{end_time}\t{seg['text'].strip()}\n"
    return output.strip()

def generate_plain_srt_from_segments(segments):
    """Generate a plain text SRT file from segments for download."""
    output = ""
    for i, seg in enumerate(segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        output += f"{i}\n{start_time} --> {end_time}\n{seg['text'].strip()}\n\n"
    return output.strip()

def generate_plain_vtt_from_segments(segments):
    """Generate a plain text VTT file from segments for download."""
    output = "WEBVTT\n\n"
    for seg in segments:
        start_time = format_time(seg["start"]).replace(",", ".")
        end_time = format_time(seg["end"]).replace(",", ".")
        output += f"{start_time} --> {end_time}\n{seg['text'].strip()}\n\n"
    return output.strip()

######################################################
# Additional utility for measuring tokens
######################################################
def calculate_token_count(text, model_name="gpt-4"):
    if not text:
        return 0
    return len(re.findall(r'\w+', text))

######################################################
# Utility to parse response segments or fallback
######################################################
def estimate_segments_from_text(full_text, audio_filepath):
    """
    Estimates segments with approximate timings if the API doesn't
    provide them. Uses mutagen to get duration, or fallback to word-based.
    """
    segments = []
    duration = 0.0
    try:
        audio_info = mutagen.File(audio_filepath)
        if audio_info and hasattr(audio_info, 'info') and hasattr(audio_info.info, 'length'):
            duration = audio_info.info.length
        else:
            raise ValueError("mutagen couldn't determine audio duration.")
    except Exception as e:
        # Fallback to average speaking rate
        word_count = len(re.findall(r'\w+', full_text))
        if word_count > 0:
            duration = word_count * 0.4
        else:
            duration = 1.0

    if not full_text or duration <= 0:
        return [{"start": 0, "end": max(duration, 1.0), "text": full_text or "[Empty Transcript]"}]

    # Try to split by sentence punctuation
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())

    # If we only get 1 big sentence or they're too big, chunk by words
    if not sentences or len(sentences) <= 1:
        words = re.findall(r'\S+', full_text)
        chunk_size = 15
        sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    total_chars = len(full_text)
    start_time = 0.0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        seg_chars = len(sentence)
        seg_duration = (seg_chars / total_chars) * duration if total_chars > 0 else 0
        seg_duration = max(seg_duration, 0.1)
        end_time = min(start_time + seg_duration, duration)
        start_time = min(start_time, end_time)

        segments.append({
            "start": start_time,
            "end": end_time,
            "text": sentence
        })
        start_time = end_time

    # Ensure the last segment ends exactly at total duration
    if segments:
        segments[-1]["end"] = duration

    return segments

######################################################
# Simple HTML → plain text converters
######################################################
def html_to_plain_text(html_content):
    if not html_content:
        return ""
    # Remove the <button> bits
    plain_text = re.sub(r'<button.*?seek-btn.*?</button>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    # Remove obvious tags
    plain_text = re.sub(r'<div class="segment-number".*?>.*?</div>', '', plain_text, flags=re.IGNORECASE | re.DOTALL)
    plain_text = re.sub(r'<div class="srt-index".*?>.*?</div>', '', plain_text, flags=re.IGNORECASE | re.DOTALL)
    plain_text = re.sub(r'<div class="srt-timing".*?>.*?</div>', '', plain_text, flags=re.IGNORECASE | re.DOTALL)
    plain_text = re.sub(r'<div class="vtt-timing".*?>.*?</div>', '', plain_text, flags=re.IGNORECASE | re.DOTALL)
    plain_text = re.sub(r'<div class="vtt-header".*?>.*?</div>', '', plain_text, flags=re.IGNORECASE | re.DOTALL)
    # Remove all remaining tags
    plain_text = re.sub(r'<[^>]+>', ' ', plain_text)
    # Collapse whitespace
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()
    return plain_text

def html_to_plain_srt(srt_html):
    if not srt_html:
        return ""
    output = ""
    regex = re.compile(
        r'<div class="transcript-segment"[^>]*>.*?'
        r'<div class="srt-index[^>]*>(\d+)</div>.*?'
        r'<div class="srt-timing[^>]*>([^<]+)</div>.*?'
        r'<div class="srt-text[^>]*>(.*?)</div>.*?'
        r'</div>', re.DOTALL | re.IGNORECASE
    )
    segments = regex.findall(srt_html)
    for index, timing, text_html in segments:
        clean_text = re.sub(r'<[^>]+>', '', text_html).strip()
        clean_timing = timing.strip()
        output += f"{index}\n{clean_timing}\n{clean_text}\n\n"
    return output.strip()

def html_to_plain_vtt(vtt_html):
    if not vtt_html:
        return "WEBVTT\n\n"
    output = ""
    header_match = re.search(r'<div class="vtt-header[^>]*>(.*?)</div>', vtt_html, re.IGNORECASE | re.DOTALL)
    if header_match:
        output += re.sub(r'<[^>]+>', '', header_match.group(1)).strip() + "\n\n"
    else:
        output += "WEBVTT\n\n"

    block_regex = re.compile(
        r'<div class="transcript-segment vtt-entry"[^>]*>.*?'
        r'<div class="vtt-timing[^>]*>([^<]+)</div>.*?'
        r'<div class="vtt-text[^>]*>(.*?)</div>.*?'
        r'</div>', re.DOTALL | re.IGNORECASE
    )
    segments = block_regex.findall(vtt_html)
    for timing, text_html in segments:
        clean_text = re.sub(r'<[^>]+>', '', text_html).strip()
        output += f"{timing.strip()}\n{clean_text}\n\n"
    return output.strip()

######################################################
# UTILITY: Build all transcript formats
######################################################
def build_all_transcript_formats(segments):
    """
    Given a list of segments with {start, end, text},
    generate every needed HTML or plaintext variant.
    """
    srt_text = generate_srt_from_segments(segments)
    default_transcript = generate_synced_transcript_from_segments(segments)
    numbered_transcript = generate_numbered_transcript_from_segments(segments)
    vtt_text = generate_vtt_from_segments(segments)
    tsv_html = generate_tsv_from_segments(segments)
    tsv_plain = generate_plain_tsv_from_segments(segments)
    plain_srt_for_download = generate_plain_srt_from_segments(segments)
    plain_vtt_for_download = generate_plain_vtt_from_segments(segments)

    return {
        "srt_text": srt_text,
        "default_transcript": default_transcript,
        "numbered_transcript": numbered_transcript,
        "vtt_text": vtt_text,
        "tsv_text": tsv_html,
        "tsv_plain": tsv_plain,
        "plain_srt": plain_srt_for_download,
        "plain_vtt": plain_vtt_for_download
    }

######################################################
# UTILITY: Store results and emit final events
######################################################
def store_and_emit_transcription_results(
    job_id, model_name, segments,
    start_time, language="auto-detected",
    token_count=None, summary_text=None,
    topics_data=None
):
    """
    1) Builds all transcript formats from the segments,
    2) Stores them in `transcriptions` global,
    3) Emits final SocketIO events for success.
    """
    elapsed = int(time.time() - start_time)
    transcript_formats = build_all_transcript_formats(segments)

    # Convert the default transcript to plain text for token counting if needed
    if token_count is None:
        plain_text = html_to_plain_text(transcript_formats["default_transcript"])
        token_count = calculate_token_count(plain_text, model_name)

    # Make sure the job dict exists
    if job_id not in transcriptions:
        transcriptions[job_id] = {}

    # Combine the transcript formats + additional fields
    transcriptions[job_id][model_name] = {
        **transcript_formats,  # SRT, VTT, TSV, etc.
        "task": "transcribe",
        "language": language,
        "token_count": token_count,
        "elapsed_time": elapsed
    }

    # If we have special fields like summary or topics, store them
    if summary_text is not None:
        transcriptions[job_id][model_name]["summary_text"] = summary_text
    if topics_data is not None:
        transcriptions[job_id][model_name]["topics_data"] = topics_data

    # Emit final "progress" and "complete" events
    socketio.emit("progress_update", {
        "job_id": job_id,
        "progress": 100,
        "message": "Transcription complete",
        "elapsed": elapsed,
        "remaining": 0
    }, room=job_id)

    socketio.emit("transcription_complete", {
        "job_id": job_id,
        "model_option": model_name,
        "task": "transcribe",
        "language": language,
        **transcript_formats,  # srt_text, default_transcript, etc.
        "token_count": token_count,
        "elapsed_time": elapsed,
        # Include summary/topics if set
        "summary_text": summary_text,
        "topics_data": topics_data
    }, room=job_id)

    print(f"{model_name} transcription job {job_id} completed successfully.")

######################################################
# UTILITY: Clean up after job done
######################################################
def cleanup_temp(job_id, filepath, model=None):
    """
    Removes a temp file if it exists, pops the cancellation flag,
    optionally handles model GPU cleanup, etc.
    """
    # If there's a model object (e.g., local Whisper), you can CPU/unload
    if model is not None:
        try:
            if hasattr(model, 'to'):
                model.to('cpu')  # Move off GPU
            del model
            print("Cleaned up model reference.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache.")
        except Exception as model_err:
            print(f"Model cleanup error: {model_err}")

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Removed temp file: {filepath}")
        except Exception as e:
            print(f"Error removing temp file {filepath}: {e}")

    cancellations.pop(job_id, None)
    print(f"Cleanup done for job {job_id}.")


######################################################
# ROUTE: index
######################################################
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_file('favicon.ico', mimetype='image/vnd.microsoft.icon')

######################################################
# Background tasks for each STT approach
######################################################

# 1) OpenAI Whisper-1
def openai_whisper1_transcription(job_id, filepath, api_key):
    start_time = time.time()
    model_name = "whisper-1"
    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Starting OpenAI {model_name} transcription",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        openai.api_key = api_key
        with open(filepath, "rb") as audio_file:
            try:
                # Some versions:
                #   openai.Audio.transcribe(model_name, audio_file, response_format="verbose_json")
                # Others might have:
                #   openai.OpenAI(api_key=...).audio.transcriptions.create(model=..., file=..., ...)
                response = openai.Audio.transcribe(
                    model=model_name,
                    file=audio_file,
                    response_format="verbose_json"
                )
            except Exception as e:
                raise Exception(f"OpenAI API error: {e}")

        if cancellations.get(job_id, False):
            return

        if not isinstance(response, dict):
            response = dict(response)

        segments = []
        if "segments" in response and response["segments"]:
            segments = response["segments"]
        elif "text" in response:
            # Fallback
            segments = estimate_segments_from_text(response["text"], filepath)
        else:
            raise Exception("No segments or text returned by whisper-1")

        language = response.get("language", "auto-detected")

        # Store & emit
        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_name,
            segments=segments,
            start_time=start_time,
            language=language
        )
    except Exception as e:
        print(f"Error in OpenAI {model_name} transcription: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_name,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, filepath)

# 2) OpenAI GPT-4o family
def openai_gpt4o_family_transcription(job_id, filepath, api_key, model_name, prompt=None):
    """
    Handles both 'gpt-4o-transcribe' and 'gpt-4o-mini-transcribe'.
    Allows optional prompt to customize transcription behavior.
    """
    start_time = time.time()
    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Starting OpenAI {model_name} transcription",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        openai.api_key = api_key
        with open(filepath, "rb") as audio_file:
            # Setup transcription parameters with or without prompt
            transcription_params = {
                "model": model_name,
                "file": audio_file,
                "response_format": "json"
            }
            
            # Only add prompt parameter if it was provided and not empty
            if prompt and prompt.strip():
                transcription_params["prompt"] = prompt
                print(f"Using prompt for {model_name} transcription: {prompt[:100]}...")
            
            try:
                # Try new OpenAI package version first (v1.0.0+)
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    response = client.audio.transcriptions.create(**transcription_params)
                except (ImportError, AttributeError):
                    # Fall back to older version (v0.x.x)
                    response = openai.Audio.transcribe(**transcription_params)

                if not isinstance(response, dict):
                    response = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
            except Exception as api_err:
                print(f"Error calling OpenAI API for {model_name}: {api_err}")
                raise Exception(f"OpenAI API Error ({model_name}): {str(api_err)}")

        if cancellations.get(job_id, False):
             return

        full_text = response.get("text", "")
        language = response.get("language", "auto-detected")
        if not full_text.strip():
            # Empty text fallback
            segments = [{
                "start": 0,
                "end": 1,
                "text": "[No transcript returned]"
            }]
        else:
            segments = estimate_segments_from_text(full_text, filepath)

        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_name,
            segments=segments,
            start_time=start_time,
            language=language
        )
    except Exception as e:
        print(f"Error in OpenAI {model_name} transcription: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_name,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, filepath)

# 3) Deepgram
def extract_segments_from_deepgram_response(result):
    """Original logic from the code that tries utterances, paragraphs, words, etc."""
    # (Unchanged logic for extracting segments from the Deepgram response)
    segments = []
    if not result or "results" not in result:
        return segments

    results_data = result["results"]
    # Prefer utterances
    if "utterances" in results_data and results_data["utterances"]:
        for utt in results_data["utterances"]:
            segments.append({
                "start": utt.get("start", 0),
                "end": utt.get("end", 0),
                "text": utt.get("transcript", "").strip()
            })
        return segments # Return after finding utterances

    # Fallback to channels + paragraphs + words, etc.
    if "channels" in results_data and len(results_data["channels"]) > 0:
        channel = results_data["channels"][0]
        if "alternatives" in channel and len(channel["alternatives"]) > 0:
            alt = channel["alternatives"][0]
            
            # First try paragraphs if available
            if "paragraphs" in alt and "paragraphs" in alt["paragraphs"]:
                for para in alt["paragraphs"]["paragraphs"]:
                    segments.append({
                        "start": para.get("start", 0),
                        "end": para.get("end", 0),
                        "text": para.get("text", "").strip()
                    })
                return segments
            
            # Then try words as a last resort
            elif "words" in alt:
                words = alt["words"]
                if not words:
                    return segments
                    
                # Group words into reasonable segments (e.g., ~10 words per segment)
                chunk_size = 10
                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i+chunk_size]
                    if chunk:
                        segments.append({
                            "start": chunk[0].get("start", 0),
                            "end": chunk[-1].get("end", 0),
                            "text": " ".join(w.get("word", "") for w in chunk).strip()
                        })
                return segments

    # If we get here with no segments, return empty
    return segments

def extract_full_text_from_deepgram(result):
    """Try to get the full transcript text from the first alternative."""
    try:
        if (result.get("results") and
            result["results"].get("channels") and
            len(result["results"]["channels"]) > 0):
            alt = result["results"]["channels"][0]["alternatives"][0]
            if "transcript" in alt:
                return alt["transcript"].strip()
    except:
        pass
    return ""

def deepgram_ai_features(api_key, full_text, do_summarize=False, do_topics=False):
    """
    Calls Deepgram's summarization/topics if requested.
    Returns (summary_text, topics_data).
    """
    summary_text, topics_data = None, None
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    data_payload = {"text": full_text}
    params = {"language": "en"}  # Summaries/Topics only supported in English
    # If we wanted to skip if not English, you'd check the language first.

    # Summarize
    if do_summarize:
        summary_params = dict(params, summarize="true")
        try:
            r = http.post("https://api.deepgram.com/v1/read",
                          params=summary_params, headers=headers,
                          json=data_payload, timeout=30)
            if r.status_code == 200:
                jr = r.json()
                summary_text = jr["results"]["summary"].get("text", None)
        except Exception as e:
            print(f"Deepgram summary error: {e}")

    # Topics
    if do_topics:
        topics_params = dict(params, topics="true")
        try:
            r = http.post("https://api.deepgram.com/v1/read",
                          params=topics_params, headers=headers,
                          json=data_payload, timeout=30)
            if r.status_code == 200:
                jr = r.json()
                topics_data = jr["results"].get("topics", None)
        except Exception as e:
            print(f"Deepgram topics error: {e}")

    return summary_text, topics_data

def deepgram_transcription(job_id, filepath, api_key, summarize_enabled=False, topics_enabled=False):
    start_time = time.time()
    model_name = "nova-3"
    summary_text = None
    topics_data = None
    detected_language = "multi"

    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Starting Deepgram {model_name} transcription",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        file_ext = os.path.splitext(filepath)[1].lower()
        content_type = mime_types_map.get(file_ext, "application/octet-stream")
        headers = {"Authorization": f"Token {api_key}", "Content-Type": content_type}

        # Stream upload directly from disk (avoid loading whole file into memory)
        # Note: keep the file open only during the request

        params = {
            "model": model_name,
            "language": "multi",
            "diarize": "true",
            "punctuate": "true",
            "paragraphs": "true",
            "utterances": "true",
            "smart_format": "true"
        }

        with open(filepath, 'rb') as f:
            response = http.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                data=f,
                timeout=120
            )
        if response.status_code != 200:
            raise Exception(f"Deepgram error: {response.status_code}, {response.text}")

        result = response.json()

        segments = extract_segments_from_deepgram_response(result)
        full_transcript_text = extract_full_text_from_deepgram(result)

        # Fallback if no segments
        if not segments and full_transcript_text:
            segments = estimate_segments_from_text(full_transcript_text, filepath)

        # Summaries/Topics if English, but skipping language check for brevity
        if full_transcript_text and (summarize_enabled or topics_enabled):
            summary_text, topics_data = deepgram_ai_features(
                api_key, full_transcript_text,
                do_summarize=summarize_enabled,
                do_topics=topics_enabled
            )

        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_name,
            segments=segments,
            start_time=start_time,
            language=detected_language,
            summary_text=summary_text,
            topics_data=topics_data
        )
    except Exception as e:
        print(f"Deepgram transcription error: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_name,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, filepath)

# 4) Gladia
def gladia_transcription(job_id, filepath, api_key):
    start_time = time.time()
    model_name = "gladia"
    detected_language = "auto-detected"

    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Starting Gladia v2 transcription",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        mime_type = mime_types_map.get(file_ext, "application/octet-stream")
        upload_headers = {"x-gladia-key": api_key}

        # Step 1) Upload
        with open(filepath, "rb") as audio_file:
            files = {"audio": (filename, audio_file, mime_type)}
            r = http.post("https://api.gladia.io/v2/upload",
                          headers=upload_headers, files=files, timeout=120)
            r.raise_for_status()
            upload_result = r.json()
            audio_url = upload_result.get("audio_url")
            if not audio_url:
                raise Exception("No audio_url in Gladia v2 upload response")

        # Step 2) Request transcription
        transcription_headers = {
            "x-gladia-key": api_key,
            "Content-Type": "application/json"
        }
        transcription_data = {
            "audio_url": audio_url,
            "language_config": {
                "languages": [],
                "code_switching": False
            },
            "translation": False,
            "diarization": True,
            "diarization_config": {
                "number_of_speakers": None,
                "min_speakers": 1,
                "max_speakers": 5,
                "enhanced": True
            },
            "subtitles": True,
            "subtitles_config": {
                "formats": ["srt", "vtt"],
                "maximum_characters_per_row": 40,
                "maximum_rows_per_caption": 2
            }
        }
        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 50,
            "message": "Processing with Gladia API",
            "elapsed": int(time.time() - start_time),
            "remaining": 0
        }, room=job_id)
        tresp = http.post("https://api.gladia.io/v2/pre-recorded",
                          headers=transcription_headers, json=transcription_data, timeout=30)
        tresp.raise_for_status()
        tr_json = tresp.json()
        transcription_id = tr_json.get("id")
        result_url = tr_json.get("result_url")
        if not transcription_id or not result_url:
            raise Exception("Failed to get transcription ID or result_url from Gladia v2 response")

        # Step 3) Poll for results
        max_retries = 60
        polling_interval = 3
        final_result = None
        for attempt in range(max_retries):
            if cancellations.get(job_id, False):
                return
            poll_resp = http.get(result_url, headers={"x-gladia-key": api_key}, timeout=30)
            if poll_resp.status_code == 200:
                prj = poll_resp.json()
                status = prj.get("status")
                if status == "done":
                    final_result = prj
                    break
            # Exponential backoff up to 15s
            time.sleep(min(15, polling_interval))
            polling_interval = min(15, polling_interval * 2)

        if not final_result or final_result.get("status") != "done":
            raise Exception("Gladia v2 transcription timed out or failed.")

        transcript_data = final_result.get("result", {})
        
        # Extract language from Gladia V2 response structure
        if ("transcription" in transcript_data and
            isinstance(transcript_data["transcription"], dict) and
            "languages" in transcript_data["transcription"] and
            transcript_data["transcription"]["languages"]):
            detected_language = transcript_data["transcription"]["languages"][0]
        
        # Step 4) Extract segments
        segments = []
        if ("transcription" in transcript_data and
            isinstance(transcript_data["transcription"], dict) and
            "utterances" in transcript_data["transcription"]):
            utterances = transcript_data["transcription"]["utterances"]
            for u in utterances:
                txt = str(u.get("text", "")).strip()
                if txt:
                    segments.append({
                        "start": u.get("start", 0),
                        "end": u.get("end", 0),
                        "text": txt
                    })
        elif ("transcription" in transcript_data and
              isinstance(transcript_data["transcription"], dict) and
              "full_transcript" in transcript_data["transcription"]):
            full_text = transcript_data["transcription"]["full_transcript"]
            segments = estimate_segments_from_text(full_text, filepath)
        else:
            # fallback to SRT
            subs = transcript_data.get("subtitles", {})
            srt_content = subs.get("srt")
            if srt_content:
                # parse SRT or fallback
                segments = []  # parse into segments if desired
                # If no parse, estimate from the entire text
                # ...
        if not segments:
            segments = [{
                "start": 0, "end": 1,
                "text": "Transcription done, but no segments found."
            }]

        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_name,
            segments=segments,
            start_time=start_time,
            language=detected_language
        )
    except Exception as e:
        print(f"Gladia v2 transcription error: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_name,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, filepath)

# 5) ElevenLabs
def extract_segments_from_elevenlabs_response(result):
    segments = []
    if not result or "words" not in result:
        return segments

    words_list = result["words"]
    current_segment = None
    max_words_per_segment = 15
    max_gap_seconds = 1.0

    for item in words_list:
        item_type = item.get("type")
        text = item.get("text", "").strip()
        start = item.get("start")
        end = item.get("end")

        if item_type == "spacing":
            continue

        if start is None or end is None:
            continue

        start_new = False
        if current_segment is None:
            start_new = True
        else:
            gap = start - current_segment["end"]
            prev_ends_sentence = current_segment["text"].endswith(('.', '!', '?'))
            if (gap > max_gap_seconds or
                current_segment["words_in_segment"] >= max_words_per_segment or
                prev_ends_sentence):
                start_new = True

        if start_new and current_segment:
            current_segment["text"] = current_segment["text"].strip()
            if current_segment["text"]:
                segments.append(current_segment)
            current_segment = None

        if start_new:
            current_segment = {
                "start": start,
                "end": end,
                "text": "",
                "words_in_segment": 0
            }

        needs_space = current_segment["text"] and not current_segment["text"].endswith('[')
        current_segment["text"] += (" " if needs_space else "") + text
        current_segment["end"] = max(current_segment["end"], end)
        if item_type == "word":
             current_segment["words_in_segment"] += 1

    if current_segment and current_segment["text"].strip():
        segments.append(current_segment)

    if not segments:
        full_text = result.get("text", "").strip()
        if full_text:
            duration = words_list[-1]["end"] if words_list else 1.0
            segments.append({
                "start": 0,
                "end": duration,
                "text": full_text
            })
    return segments

def elevenlabs_transcription(job_id, filepath, api_key, language_code=None):
    start_time = time.time()
    model_name = "scribe-v1"
    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Starting ElevenLabs {model_name} transcription",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        # Validate API key format
        if not api_key or len(api_key.strip()) < 10:
            raise Exception("ElevenLabs API key is missing or too short")

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 50,
            "message": "Processing with ElevenLabs API",
            "elapsed": int(time.time() - start_time),
            "remaining": 0
        }, room=job_id)

        # Use direct HTTP API call (SDK has model_id mapping issues)
        # This matches exactly how the frontend direct API call works
        try:
            with open(filepath, "rb") as audio_file:
                # Prepare form data exactly like frontend
                files = {'file': audio_file}
                data = {
                    'model_id': 'scribe_v1',
                    'output_format': 'json',
                    'response_format': 'verbose_json'
                }
                
                # Only add language_code if provided
                if language_code and language_code.strip():
                    data['language_code'] = language_code.strip()
                
                headers = {'xi-api-key': api_key.strip()}
                
                # Use fresh requests call instead of global session
                response = requests.post(
                    'https://api.elevenlabs.io/v1/speech-to-text',
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300
                )
                
                if not response.ok:
                    error_text = response.text
                    
                    if response.status_code == 401:
                        raise Exception("Invalid ElevenLabs API key. Please check your API key in the settings.")
                    elif response.status_code == 400:
                        # Parse the error for better messaging
                        try:
                            error_json = response.json()
                            error_msg = error_json.get('message', error_text)
                        except:
                            error_msg = error_text
                        raise Exception(f"ElevenLabs API error: {error_msg}")
                    else:
                        raise Exception(f"ElevenLabs API error: {response.status_code} - {error_text}")
                
                result = response.json()
                
        except Exception as api_error:
            error_msg = str(api_error)
            print(f"ElevenLabs API error details: {error_msg}")
            
            if "invalid_api_key" in error_msg.lower() or "401" in error_msg:
                raise Exception("Invalid ElevenLabs API key. Please check your API key in the settings.")
            else:
                raise Exception(f"ElevenLabs API error: {error_msg}")

        detected_language = result.get("language_code", "auto-detected")
        segments = extract_segments_from_elevenlabs_response(result)
        if not segments:
            segments = [{
                "start": 0, "end": 1,
                "text": "Error: no segments parsed."
            }]

        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_name,
            segments=segments,
            start_time=start_time,
            language=detected_language
        )
    except Exception as e:
        print(f"Error in ElevenLabs transcription: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_name,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, filepath)

######################################################
# 6) Local Transcription (Whisper)
######################################################
def fast_transcription(job_id, file_path, model_option):
    start_time = time.time()
    local_model_key = model_option
    model = None

    try:
        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 10,
            "message": f"Loading {local_model_key} model...",
            "elapsed": 0,
            "remaining": 0
        }, room=job_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Example of a custom mapping
        if local_model_key == "turbo":
            local_model_key = "large-v2"  # Example mapping

        model = whisper.load_model(local_model_key, device=device)

        if cancellations.get(job_id, False):
            return

        socketio.emit("progress_update", {
            "job_id": job_id,
            "progress": 30,
            "message": f"Transcribing with {local_model_key}...",
            "elapsed": int(time.time() - start_time),
            "remaining": 0
        }, room=job_id)

        options = {
            "task": "transcribe",
            "word_timestamps": True
        }
        if torch.cuda.is_available():
            options["fp16"] = True

        # Reduce Python overhead and enable better kernels where possible
        with torch.inference_mode():
            result = model.transcribe(file_path, **options)
        segments = result.get("segments", [])
        detected_language = result.get("language", "auto-detected")

        # Basic post-processing or fallback
        if not segments:
            full_text = result.get("text", "").strip()
            if full_text:
                segments = estimate_segments_from_text(full_text, file_path)
            else:
                segments = [{
                    "start": 0, "end": 1,
                    "text": "[No speech detected]"
                }]

        store_and_emit_transcription_results(
            job_id=job_id,
            model_name=model_option,  # keep user-chosen name
            segments=segments,
            start_time=start_time,
            language=detected_language
        )
    except Exception as e:
        print(f"Error in local transcription: {e}")
        if not cancellations.get(job_id, False):
            socketio.emit("transcription_failed", {
                "job_id": job_id,
                "model_option": model_option,
                "error": str(e)
            }, room=job_id)
    finally:
        cleanup_temp(job_id, file_path, model=model)

######################################################
# ROUTES (front-end triggers)
######################################################
@app.route('/transcribe', methods=['POST'])
def transcribe_api():
    # For the "whisper-1" direct route
    api_key = request.form.get('api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')
    if not api_key or not file or not job_id:
        return jsonify({"error": "API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(openai_whisper1_transcription, job_id, filepath, api_key)
    return jsonify({"job_id": job_id, "status": "started", "model_option": "whisper-1"})

@app.route('/transcribe_openai_gpt4o', methods=['POST'])
def transcribe_openai_gpt4o():
    # "gpt-4o-transcribe"
    api_key = request.form.get('api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')
    prompt = request.form.get('prompt')  # Get prompt from request
    model_option = "gpt-4o-transcribe"
    
    if not api_key or not file or not job_id:
        return jsonify({"error": "OpenAI API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(
        openai_gpt4o_family_transcription, 
        job_id, 
        filepath, 
        api_key, 
        model_option,
        prompt  # Pass prompt to the transcription function
    )
    return jsonify({"job_id": job_id, "status": "started", "model_option": model_option})

@app.route('/transcribe_openai_gpt4o_mini', methods=['POST'])
def transcribe_openai_gpt4o_mini():
    # "gpt-4o-mini-transcribe"
    api_key = request.form.get('api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')
    prompt = request.form.get('prompt')  # Get prompt from request
    model_option = "gpt-4o-mini-transcribe"
    
    if not api_key or not file or not job_id:
        return jsonify({"error": "OpenAI API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(
        openai_gpt4o_family_transcription, 
        job_id, 
        filepath, 
        api_key, 
        model_option,
        prompt  # Pass prompt to the transcription function
    )
    return jsonify({"job_id": job_id, "status": "started", "model_option": model_option})

@app.route('/transcribe_deepgram', methods=['POST'])
def transcribe_deepgram_route():
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
def transcribe_gladia_route():
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

@app.route('/transcribe_elevenlabs', methods=['POST'])
def transcribe_elevenlabs_route():
    api_key = request.form.get('elevenlabs_api_key')
    file = request.files.get('file')
    job_id = request.form.get('job_id')
    if not api_key or not file or not job_id:
        return jsonify({"error": "ElevenLabs API key, file, and job_id are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(elevenlabs_transcription, job_id, filepath, api_key)
    return jsonify({"job_id": job_id, "status": "started", "model_option": "scribe-v1"})

@app.route('/transcribe_local', methods=['POST'])
def transcribe_local():
    file = request.files.get('file')
    model_option = request.form.get('model_option', 'tiny')
    job_id = request.form.get('job_id')
    if not file or not job_id or not model_option:
        return jsonify({"error": "File, job_id, and model_option are required."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    file.save(filepath)

    cancellations[job_id] = False
    socketio.start_background_task(fast_transcription, job_id, filepath, model_option)
    return jsonify({"job_id": job_id, "status": "started", "model_option": model_option})

######################################################
# SocketIO: Cancel + Join
######################################################
@socketio.on("cancel_transcription")
def handle_cancel_transcription(data):
    job_id = data.get("job_id")
    if job_id:
        print(f"Cancellation requested for job: {job_id}")
        cancellations[job_id] = True
        # First emit that we're cancelling
        emit("transcription_cancelling", {
            "job_id": job_id,
            "message": "Cancellation request received..."
        }, room=job_id)
        # Then immediately emit the cancelled event
        emit("transcription_cancelled", {
            "job_id": job_id,
            "message": "Transcription cancelled by user"
        }, room=job_id)

@socketio.on("join")
def on_join(data):
    job_id = data.get("job_id")
    sid = request.sid
    if job_id:
        join_room(job_id)
        print(f"Socket client {sid} joined room: {job_id}")

######################################################
# Download endpoint
######################################################
@app.route('/download_srt', methods=['POST'])
def download_srt():
    output_type = request.form.get('output_type', 'srt').lower()
    job_id = request.form.get('job_id')
    model = request.form.get('model', '')
    print(f"Download requested: job={job_id}, model={model}, type={output_type}")

    if not job_id or job_id not in transcriptions:
        return jsonify({"error": "Job ID not found or no transcriptions for this ID."}), 400

    model_results = None
    if model and model in transcriptions[job_id]:
        model_results = transcriptions[job_id][model]
    else:
        # If only one model is present, pick it automatically
        available_models = list(transcriptions[job_id].keys())
        if len(available_models) == 1:
            model = available_models[0]
            model_results = transcriptions[job_id][model]
        else:
            return jsonify({"error": f"Multiple models found for {job_id} ({', '.join(available_models)}). Specify one."}), 400

    output_text = ""
    ext = "txt"

    try:
        if output_type == "srt":
            ext = "srt"
            if "plain_srt" in model_results and model_results["plain_srt"]:
                 output_text = model_results["plain_srt"]
            else:
                # Convert from HTML
                 output_text = html_to_plain_srt(model_results.get("srt_text", ""))
        elif output_type == "vtt":
            ext = "vtt"
            output_text = html_to_plain_vtt(model_results.get("vtt_text", ""))
        elif output_type == "tsv":
            ext = "tsv"
            if "tsv_plain" in model_results and model_results["tsv_plain"]:
                 output_text = model_results["tsv_plain"]
            else:
                # fallback
                output_text = html_to_plain_text(model_results.get("default_transcript", ""))
                ext = "txt"
        else: # Correctly aligned with the elif blocks
            # default to plain txt
            ext = "txt"
            output_text = html_to_plain_text(model_results.get("default_transcript", ""))

        if not output_text:
            print("Warning: output text is empty, returning empty file.")

        download_filename = f"{model}_{job_id}.{ext}"
        output_file = os.path.join(app.config['TEMP_FOLDER'], f"download_{job_id}_{model}.{ext}")
        with open(output_file, "w", encoding="utf-8") as f:
             f.write(output_text)

        return send_file(output_file, as_attachment=True, download_name=download_filename)
    except Exception as e:
        print(f"Error generating file for download: {e}")
        return jsonify({"error": f"Failed to generate file: {str(e)}"}), 500

######################################################
# MAIN
######################################################
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
