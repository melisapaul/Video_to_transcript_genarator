import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Load .env
load_dotenv()

# If a bundled ffmpeg is present at `./ffmpeg/bin`, prepend it to the PATH so
# subprocess calls (ffmpeg, ffprobe, yt-dlp) resolve the local executables on
# Windows and other platforms.
try:
    REPO_ROOT = Path(__file__).parent
except NameError:
    REPO_ROOT = Path.cwd()
_local_ffmpeg_dir = REPO_ROOT / "ffmpeg" / "bin"
if _local_ffmpeg_dir.exists():
    os.environ["PATH"] = str(_local_ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")

# Config
ASR_MODEL = os.getenv("ASR_MODEL", "small")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    import whisper
except Exception as e:
    whisper = None
    whisper_import_error = e

try:
    from transformers import pipeline
except Exception as e:
    pipeline = None
    transformers_import_error = e

try:
    import torch
    _torch_available = True
except Exception:
    _torch_available = False


# Cache helpers for heavy models so they are loaded once per session.
try:
    @st.cache_resource
    def get_whisper_model_cached(model_name: str):
        return whisper.load_model(model_name)

    @st.cache_resource
    def get_summarizer_cached(model_name: str):
        device = 0 if _torch_available and torch.cuda.is_available() else -1
        return pipeline("summarization", model=model_name, device=device)
except Exception:
    # If Streamlit cache_resource isn't available (older versions), fall back to no-op wrappers
    def get_whisper_model_cached(model_name: str):
        return whisper.load_model(model_name)

    def get_summarizer_cached(model_name: str):
        device = 0 if _torch_available and _torch_available and (hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available()) else -1
        return pipeline("summarization", model=model_name, device=device)


# ---------------- Helpers -----------------
def run_cmd(cmd: List[str], check=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip(), proc.stderr.strip()


def check_binaries():
    """Verify required external binaries are available: ffmpeg, ffprobe, yt-dlp.
    If any are missing, return a friendly message string; otherwise return None.
    """
    import shutil as _sh

    missing = []
    for name in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not _sh.which(name):
            missing.append(name)

    if missing:
        return (
            f"Missing required binaries: {', '.join(missing)}.\n"
            "Install them or place a bundled `ffmpeg/bin` in the repo root."
        )
    return None


def download_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "yt_audio.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "-o",
        out_template,
        url,
    ]
    run_cmd(cmd)

    for ext in ("mp3", "m4a", "wav", "webm", "ogg"):
        f = out_dir / f"yt_audio.{ext}"
        if f.exists():
            return f

    raise RuntimeError("Audio download failed.")


def convert_to_wav(input_path: Path, out_path: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(out_path),
    ]
    run_cmd(cmd)


def load_whisper(model_name: str):
    if whisper is None:
        raise RuntimeError("Whisper not installed. pip install openai-whisper")
    return get_whisper_model_cached(model_name)


def transcribe(model, wav_path: Path, language=None):
    options = {}
    if language:
        options["language"] = language
        options["task"] = "transcribe"
    return model.transcribe(str(wav_path), **options)


def shift_segments(segments: List[Dict[str, Any]], offset: float) -> List[Dict[str, Any]]:
    out = []
    for seg in segments:
        seg_copy = dict(seg)
        seg_copy["start"] = seg_copy.get("start", 0.0) + offset
        seg_copy["end"] = seg_copy.get("end", 0.0) + offset
        out.append(seg_copy)
    return out


def pretty_transcript(result: Dict[str, Any]) -> str:
    lines = []
    for seg in result.get("segments", []):
        s, e = seg["start"], seg["end"]
        txt = seg["text"].strip()

        def fmt(t):
            return f"{int(t//60):02d}:{int(t%60):02d}"

        lines.append(f"[{fmt(s)} - {fmt(e)}] {txt}")

    if not lines:
        return result.get("text", "")
    return "\n".join(lines)


def chunk_text(text, max_chars=2500):
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s)
            cur_len += len(s) + 1
        else:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarize(text: str, model_name: str):
    if pipeline is None:
        raise RuntimeError("Transformers not installed.")

    summarizer = get_summarizer_cached(model_name)

    chunks = chunk_text(text)
    outs = [summarizer(c, max_length=150, min_length=30, truncation=True)[0]["summary_text"] for c in chunks]

    if len(outs) == 1:
        return outs[0]

    final = summarizer(" ".join(outs), max_length=200, min_length=40, truncation=True)[0]["summary_text"]
    return final


# ---------------- UI -----------------
st.title("LectureNotes — Auto Transcript & Summary")

input_type = st.radio("Choose Input", ["YouTube URL", "Upload File"])

yt_url = None
file_upload = None

if input_type == "YouTube URL":
    yt_url = st.text_input("Enter YouTube URL")
else:
    file_upload = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "mp4", "mkv", "webm", "m4a"])

language = st.text_input("Language (optional)", "")
asr_model = st.text_input("Whisper Model", ASR_MODEL)
sum_model = st.text_input("Summarizer Model", SUMMARIZER_MODEL)

if st.button("Process"):
    try:
        temp = Path(tempfile.mkdtemp())
        try:
            # 1. Source audio
            with st.spinner("Sourcing audio..."):
                if input_type == "YouTube URL":
                    if not yt_url:
                        st.error("Please enter a URL")
                        st.stop()
                    audio_src = download_audio(yt_url, temp)
                else:
                    if not file_upload:
                        st.error("Please upload a file")
                        st.stop()
                    audio_src = temp / file_upload.name
                    with open(audio_src, "wb") as f:
                        f.write(file_upload.read())


            # 2. Convert to wav
            wav_path = temp / "audio.wav"
            with st.spinner("Converting to WAV (ffmpeg)..."):
                convert_to_wav(audio_src, wav_path)

            # 3. Progressive transcription: split WAV into segments and transcribe each
            with st.spinner("Loading ASR model..."):
                model = load_whisper(asr_model)

            # Ensure required binaries exist
            missing_msg = check_binaries()
            if missing_msg:
                st.error(missing_msg)
                st.stop()

            seg_dir = temp / "segments"
            seg_dir.mkdir(parents=True, exist_ok=True)

            # Split into 60 second segments to allow progressive transcription
            split_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-f",
                "segment",
                "-segment_time",
                "60",
                "-reset_timestamps",
                "1",
                str(seg_dir / "seg_%03d.wav"),
            ]
            run_cmd(split_cmd)

            all_segments: List[Dict[str, Any]] = []
            texts: List[str] = []
            transcript_area = st.empty()

            seg_files = sorted(seg_dir.glob("seg_*.wav"))
            for idx, seg_f in enumerate(seg_files):
                with st.spinner(f"Transcribing segment {idx+1}/{len(seg_files)}..."):
                    seg_result = transcribe(model, seg_f, language or None)

                # shift timestamps by segment index * 60 (segment length)
                offset = idx * 60
                segs = seg_result.get("segments", [])
                shifted = shift_segments(segs, offset)
                all_segments.extend(shifted)
                texts.append(seg_result.get("text", ""))

                # update UI progressively
                transcript_partial = pretty_transcript({"segments": all_segments})
                transcript_area.text_area("Transcript (partial)", transcript_partial, height=300)

            # final transcript
            transcript = pretty_transcript({"segments": all_segments})

            # Save
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(OUTPUT_DIR / "transcript.txt", "w", encoding="utf8") as f:
                f.write(transcript)

            st.subheader("Transcript")
            st.text_area("Transcript", transcript, height=300)

            # 4. Summary
            full_text = " \n".join(texts)
            with st.spinner("Summarizing transcription..."):
                summary = summarize(full_text, sum_model)
                with open(OUTPUT_DIR / "summary.md", "w", encoding="utf8") as f:
                    f.write(summary)

            st.subheader("Summary")
            st.write(summary)

            st.download_button("Download Transcript", transcript, "transcript.txt")
            st.download_button("Download Summary", summary, "summary.md")
        finally:
            try:
                shutil.rmtree(temp)
            except Exception:
                pass

    except Exception as e:
        st.error(str(e))
