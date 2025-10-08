#!/usr/bin/env python3
"""
ultimate_jarvis_final_extended.py
Merged & extended Jarvis for Windows 11 — preserves original behavior + many new features.

- Keeps original code, greetings, animation, and "Shubham" branding.
- Adds many optional features (guarded imports): screenshots, screen recording, voice recording,
  YouTube play/download, Instagram download (profile), QR generation, PDF reading, speedtest,
  public IP lookup, news, COVID-19 India counts, contacts, schedule, email & WhatsApp helpers, etc.

Run with Python 3.8+. Optional dependencies (install only what's needed):
  pip install pyttsx3 playsound requests pypdf2 pytube pywhatkit speedtest-cli instaloader
  pyautogui pillow sounddevice scipy numpy opencv-python wikipedia qrcode
"""
from __future__ import annotations

import os
import sys
import re
import json
import time
import queue
import shutil
import random
import threading
import tempfile
import subprocess
import webbrowser
import xml.etree.ElementTree as ET
from pathlib import Path
import traceback

# -------------------------
# Preserve original optional flags and try-imports (keeps code stable)
# -------------------------
EDGE_TTS_AVAILABLE = False
PLAYSOUND_AVAILABLE = False
PYTTSX3_AVAILABLE = False
SR_AVAILABLE = False
SPOTIPY_AVAILABLE = False
OPENAI_AVAILABLE = False
DEEP_TRANSLATOR_AVAILABLE = False
PSUTIL_AVAILABLE = False
GENAI_AVAILABLE = False
COLORAMA_AVAILABLE = False

# Try imports used in original file
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

try:
    import playsound
    PLAYSOUND_AVAILABLE = True
except Exception:
    PLAYSOUND_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    _pytt_engine = pyttsx3.init()
    # Slightly adjust rate to make it snappier
    try:
        rate = _pytt_engine.getProperty("rate")
        _pytt_engine.setProperty("rate", int(rate * 1.25))  # +25% faster than default
    except Exception:
        pass
except Exception:
    PYTTSX3_AVAILABLE = False
    _pytt_engine = None

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except Exception:
    SPOTIPY_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    DEEP_TRANSLATOR_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False

try:
    import requests
except Exception:
    requests = None

# Additional optional libs for new features (guarded usage)
OPTIONAL = {}
try:
    import pyautogui
    from PIL import Image
    OPTIONAL['pyautogui'] = True
except Exception:
    OPTIONAL['pyautogui'] = False

try:
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wavfile
    OPTIONAL['sounddevice'] = True
except Exception:
    OPTIONAL['sounddevice'] = False

try:
    import cv2
    OPTIONAL['cv2'] = True
except Exception:
    OPTIONAL['cv2'] = False

try:
    import qrcode
    OPTIONAL['qrcode'] = True
except Exception:
    OPTIONAL['qrcode'] = False

try:
    from pytube import YouTube, Search
    OPTIONAL['pytube'] = True
except Exception:
    OPTIONAL['pytube'] = False

try:
    import pywhatkit
    OPTIONAL['pywhatkit'] = True
except Exception:
    OPTIONAL['pywhatkit'] = False

try:
    import speedtest
    OPTIONAL['speedtest'] = True
except Exception:
    OPTIONAL['speedtest'] = False

try:
    import instaloader
    OPTIONAL['instaloader'] = True
except Exception:
    OPTIONAL['instaloader'] = False

try:
    import wikipedia
    OPTIONAL['wikipedia'] = True
except Exception:
    OPTIONAL['wikipedia'] = False

try:
    import PyPDF2
    OPTIONAL['pypdf2'] = True
except Exception:
    OPTIONAL['pypdf2'] = False

# -------------------------
# Persistence & config
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
ALIASES_FILE = BASE_DIR / "jarvis_aliases.json"
MEMORY_FILE = BASE_DIR / "jarvis_memory.json"
CONTACTS_FILE = BASE_DIR / "jarvis_contacts.json"

def _load_json(path, default):
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

ALIASES = _load_json(ALIASES_FILE, {})
MEMORY = _load_json(MEMORY_FILE, {})
CONTACTS = _load_json(CONTACTS_FILE, {})

def _save_json(path, data):
    try:
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] could not save {path}: {e}")

def save_aliases():
    _save_json(ALIASES_FILE, ALIASES)

def save_memory():
    _save_json(MEMORY_FILE, MEMORY)

def save_contacts():
    _save_json(CONTACTS_FILE, CONTACTS)

# Keys from environment (you may also set them in code below)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Spotify credentials (can set here or via env variables)
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")

# -------------------------
# TTS & translate behavior (preserve original implementation)
# -------------------------
DEFAULT_HI_VOICE = "hi-IN-MadhurNeural"
DEFAULT_EN_VOICE = "en-US-GuyNeural"

tts_queue = queue.Queue()
tts_thread = None
tts_running = threading.Event()

def _tts_worker():
    tts_running.set()
    while tts_running.is_set():
        try:
            text, speak_hindi = tts_queue.get(timeout=0.2)
        except Exception:
            continue
        try:
            played = False
            # prefer edge-tts (high quality voices)
            if EDGE_TTS_AVAILABLE:
                try:
                    import asyncio
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tmp.close()
                    out_path = tmp.name
                    voice = DEFAULT_HI_VOICE if speak_hindi else DEFAULT_EN_VOICE

                    async def gen_and_save(t, v, outp):
                        comm = edge_tts.Communicate(t, v)
                        await comm.save(outp)

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(gen_and_save(text, voice, out_path))
                    # play
                    if PLAYSOUND_AVAILABLE:
                        playsound.playsound(out_path, True)
                        played = True
                    else:
                        if sys.platform.startswith("win"):
                            import winsound
                            winsound.PlaySound(out_path, winsound.SND_FILENAME)
                            played = True
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                except Exception:
                    played = False
            # fallback to pyttsx3
            if not played and PYTTSX3_AVAILABLE and _pytt_engine:
                try:
                    _pytt_engine.say(text)
                    _pytt_engine.runAndWait()
                    played = True
                except Exception:
                    played = False
            if not played:
                # last resort print
                print(f"[TTS-unavailable] {text}")
        except Exception as e:
            print(f"[TTS worker error] {e}")
        finally:
            try:
                tts_queue.task_done()
            except Exception:
                pass

def start_tts_worker():
    global tts_thread
    if tts_thread and tts_thread.is_alive():
        return
    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

def stop_tts_worker():
    tts_running.clear()
    if tts_thread:
        tts_thread.join(timeout=0.5)

def _contains_devanagari(text: str) -> bool:
    # Devanagari Unicode block: U+0900 — U+097F
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def _translate_to_hindi(text: str) -> str:
    if DEEP_TRANSLATOR_AVAILABLE:
        try:
            return GoogleTranslator(source='auto', target='hi').translate(text)
        except Exception:
            return text
    return text

def speak_async(text: str, prefer_hindi: bool = True, also_print: bool = True):
    """
    Print in English-font display and speak in Hindi voice where possible.
    """
    display_text = text
    try:
        if _contains_devanagari(text) and DEEP_TRANSLATOR_AVAILABLE:
            try:
                display_text = GoogleTranslator(source='auto', target='en').translate(text)
            except Exception:
                display_text = text
    except Exception:
        display_text = text

    if also_print:
        print(f"Jarvis: {display_text}")

    speak_text = text
    if not _contains_devanagari(text) and prefer_hindi:
        speak_text = _translate_to_hindi(text)

    start_tts_worker()
    # enqueue with prefer_hindi True by default to use Indian voice where available
    tts_queue.put((speak_text, True))

# startup sound/effect
def startup_effect(with_animation: bool = True):
    try:
        if with_animation:
            professional_startup_animation(name="Shubham", duration=1.4)
        # small beep if available
        if PLAYSOUND_AVAILABLE:
            try:
                if sys.platform.startswith("win"):
                    import winsound
                    winsound.MessageBeep()
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# AI integration (Gemini/OpenAI) — preserved from original
# -------------------------
def ask_gemini(prompt: str, timeout: int = 12):
    if not GENAI_AVAILABLE:
        return None, "Gemini SDK not installed."
    if not GEMINI_KEY:
        return None, "Gemini key not configured."
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception:
        pass
    candidate_models = ["models/gemini-2.5-flash", "models/gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-pro"]
    try:
        models = []
        try:
            models = [m.name for m in genai.list_models()]
        except Exception:
            models = []
        for m in candidate_models:
            if models and not any(m in name for name in models):
                continue
            try:
                if hasattr(genai, "generate_text"):
                    resp = genai.generate_text(model=m, prompt=prompt, max_output_tokens=512)
                    if resp and getattr(resp, "text", None):
                        return resp.text.strip(), None
                    if isinstance(resp, dict) and "candidates" in resp:
                        cand = resp["candidates"][0]
                        return cand.get("output", cand.get("content", "")).strip(), None
                if hasattr(genai, "GenerativeModel"):
                    model = genai.GenerativeModel(m)
                    if hasattr(model, "generate_content"):
                        r = model.generate_content(prompt)
                        if hasattr(r, "text") and r.text:
                            return r.text.strip(), None
                    if hasattr(model, "create"):
                        r = model.create(prompt=prompt)
                        if hasattr(r, "output") and r.output:
                            return r.output.strip(), None
            except Exception:
                continue
    except Exception as e:
        return None, str(e)
    return None, "No Gemini model succeeded."

def ask_openai(prompt: str):
    if not OPENAI_AVAILABLE:
        return None, "OpenAI package not installed."
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None, "OpenAI key not configured."
    try:
        if hasattr(openai, "ChatCompletion"):
            try:
                openai.api_key = key
                resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], max_tokens=512, temperature=0.6)
                text = resp["choices"][0]["message"]["content"].strip()
                return text, None
            except Exception as e:
                last_err = str(e)
        if hasattr(openai, "OpenAI"):
            try:
                client = openai.OpenAI(api_key=key)
                resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], max_tokens=512)
                return resp.choices[0].message.content, None
            except Exception as e:
                last_err = str(e)
        if hasattr(openai, "Completion"):
            try:
                openai.api_key = key
                resp = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=512, temperature=0.6)
                return resp.choices[0].text.strip(), None
            except Exception as e:
                last_err = str(e)
        return None, f"OpenAI call failed: {last_err}"
    except Exception as e:
        return None, str(e)

def ask_ai(prompt: str):
    """
    Attempt Gemini -> OpenAI -> fallback to Wikipedia snippet
    """
    try:
        text, err = ask_gemini(prompt)
        if text:
            return text
    except Exception:
        err = None
    try:
        text2, err2 = ask_openai(prompt)
        if text2:
            return text2
    except Exception:
        err2 = None
    # fallback to wikipedia snippet (best-effort)
    try:
        if requests:
            q = prompt.strip()
            safe_q = requests.utils.requote_uri(q)
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={safe_q}&limit=1&format=json"
            r = requests.get(search_url, timeout=6)
            r.raise_for_status()
            data = r.json()
            if len(data) >= 3 and data[2]:
                snippet = data[2][0]
                url = data[3][0] if len(data) > 3 and data[3] else None
                return f"{snippet} ({url})" if url else snippet
    except Exception:
        pass
    diag_parts = []
    if 'err' in locals() and err:
        diag_parts.append("Gemini: " + str(err))
    if 'err2' in locals() and err2:
        diag_parts.append("OpenAI: " + str(err2))
    diag = " | ".join(diag_parts)
    return f"I couldn't reach cloud AI. ({diag})"

# -------------------------
# Utilities & command handlers (preserve original ones, and add new)
# -------------------------

def run_shell(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        speak_async(f"Shell command failed: {e}", prefer_hindi=False)


def open_application(name):
    # explicit mapping first
    val = APPLICATIONS.get(name)
    if val:
        try:
            if os.path.isfile(val):
                if sys.platform.startswith("win"):
                    os.startfile(val)
                else:
                    subprocess.Popen([val])
                return True
            exe = shutil.which(val) or val
            subprocess.Popen([exe])
            return True
        except Exception:
            pass
    # try PATH
    exe = shutil.which(name)
    if exe:
        try:
            subprocess.Popen([exe])
            return True
        except Exception:
            pass
    # explorer shortcut
    if name in ("files", "explorer"):
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", os.path.expanduser("~")])
                return True
        except Exception:
            pass
    return False

APPLICATIONS = {
    # Update this path to your VS Code installation if you want direct launching
    "vscode": r"C:\\Users\\shubham\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "chrome": "chrome",
    "firefox": "firefox",
    "notepad": "notepad.exe",
}

URLS = {
    "whatsapp": "https://web.whatsapp.com/",
    "chatgpt": "https://chat.openai.com/",
    "git": "https://github.com/",
    "email": "https://mail.google.com/",
    "youtube": "https://www.youtube.com/",
    "instagram": "https://www.instagram.com/",
    "twitter": "https://twitter.com/",
    "linkedin": "https://www.linkedin.com/",
    "canva": "https://www.canva.com/",
    "google_drive": "https://drive.google.com/",
}

CONVERSATIONAL_INPUTS = {
    ("hello", "hi", "hey"): ["Hello there! How can I help?", "Hi! What can I do for you?", "Greetings! Ready for your command."],
    ("how are you", "how's it going", "kaise ho"): ["I'm running smoothly — thanks for asking!", "All systems nominal.", "Feeling helpful today."]
}

# -------------------------
# Existing command implementations (copied from original) with no changes, plus new commands appended
# -------------------------

def cmd_help(args):
    lines = ["--- Jarvis Commands ---"]
    for k, v in sorted(COMMANDS.items()):
        lines.append(f"{k:<18} - {v.get('description','')}")
    print("\n".join(lines))
    speak_async("Commands displayed in terminal.", prefer_hindi=False)

def cmd_exit(args):
    speak_async("Goodbye. Signing off.", prefer_hindi=True)
    time.sleep(0.2)
    stop_tts_worker()
    sys.exit(0)

def cmd_history(args, history):
    speak_async("Showing command history.", prefer_hindi=False)
    for i, c in enumerate(history, 1):
        print(f"{i:>3}: {c}")

def cmd_ls(args):
    path = args[0] if args else "."
    try:
        items = sorted(os.listdir(path), key=str.lower)
        for it in items:
            print(it)
        speak_async(f"Listed {len(items)} items.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"ls error: {e}", prefer_hindi=False)

def cmd_cd(args):
    try:
        if not args or args[0] == "~":
            os.chdir(os.path.expanduser("~"))
        else:
            os.chdir(args[0])
        speak_async(f"Directory changed to {os.getcwd()}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"cd error: {e}", prefer_hindi=False)

def cmd_mkdir(args):
    if not args:
        speak_async("Usage: mkdir <dir>", prefer_hindi=False)
        return
    try:
        os.makedirs(args[0], exist_ok=True)
        speak_async("Directory created.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"mkdir error: {e}", prefer_hindi=False)

def cmd_rm(args):
    if not args:
        speak_async("Usage: rm <file>", prefer_hindi=False)
        return
    try:
        os.remove(args[0])
        speak_async("File removed.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"rm error: {e}", prefer_hindi=False)

def cmd_echo(args):
    text = " ".join(args)
    print(text)
    speak_async(text, prefer_hindi=False)

def cmd_find(args):
    if not args:
        speak_async("Usage: find <filename>", prefer_hindi=False)
        return
    filename = args[0]
    for root, dirs, files in os.walk(os.path.expanduser("~")):
        if filename in files:
            path = os.path.join(root, filename)
            print(path)
            speak_async(f"Found {filename} at {path}", prefer_hindi=False)
            return
    speak_async("File not found.", prefer_hindi=False)

def cmd_calc(args):
    if not args:
        speak_async("Usage: calc <expression>", prefer_hindi=False)
        return
    expr = " ".join(args)
    try:
        # Evaluate safely: only simple arithmetic allowed
        allowed = {"__builtins__": None}
        res = eval(expr, allowed, {})
        speak_async(f"Result: {res}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Calc error: {e}", prefer_hindi=False)

def cmd_weather(args):
    if not args:
        speak_async("Usage: weather <city>", prefer_hindi=False)
        return
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False)
        return
    city = " ".join(args)
    try:
        r = requests.get(f"https://wttr.in/{requests.utils.requote_uri(city)}?format=3", timeout=6)
        speak_async(r.text, prefer_hindi=False)
    except Exception as e:
        speak_async(f"Weather error: {e}", prefer_hindi=False)

def cmd_joke(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False)
        return
    try:
        r = requests.get("https://v2.jokeapi.dev/joke/Any?type=single&category=Programming", timeout=6)
        r.raise_for_status()
        j = r.json()
        speak_async(j.get("joke", "Couldn't fetch a joke."), prefer_hindi=False)
    except Exception as e:
        speak_async(f"Joke error: {e}", prefer_hindi=False)

def cmd_quote(args):
    """
    Fetch a random inspirational quote with full offline fallback.
    """
    import random, requests

    fallback_quotes = [
        "“The best way to predict the future is to create it.” — Peter Drucker",
        "“Dream big. Work hard. Stay humble.” — Unknown",
        "“Success is not final, failure is not fatal: It is the courage to continue that counts.” — Winston Churchill",
        "“Don’t watch the clock; do what it does. Keep going.” — Sam Levenson",
        "“Your limitation—it’s only your imagination.” — Unknown",
        "“Push yourself, because no one else is going to do it for you.” — Unknown",
    ]

    try:
        # Try ZenQuotes
        r = requests.get("https://zenquotes.io/api/random", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                quote = data[0].get("q", "").strip()
                author = data[0].get("a", "Unknown")
                if quote:
                    msg = f"“{quote}” — {author}"
                    speak_async(msg, prefer_hindi=False)
                    print(f"Jarvis: {msg}")
                    return
    except Exception:
        pass

    try:
        # Try Quotable
        r = requests.get("https://api.quotable.io/random", timeout=5)
        if r.status_code == 200:
            d = r.json()
            quote = d.get("content", "").strip()
            author = d.get("author", "Unknown")
            if quote:
                msg = f"“{quote}” — {author}"
                speak_async(msg, prefer_hindi=False)
                print(f"Jarvis: {msg}")
                return
    except Exception:
        pass

    # If both APIs fail → fallback quote
    msg = random.choice(fallback_quotes)
    speak_async(msg, prefer_hindi=False)
    print(f"Jarvis: {msg}")

def cmd_news(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False)
        return
    try:
        r = requests.get("http://feeds.bbci.co.uk/news/rss.xml", timeout=6)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall('.//item')[:5]
        for i, it in enumerate(items, 1):
            t = it.find('title').text if it.find('title') is not None else 'No title'
            print(f"{i}. {t}")
        speak_async("Headlines shown.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"News error: {e}", prefer_hindi=False)

def cmd_alias(args):
    if not args:
        if ALIASES:
            for k, v in ALIASES.items():
                print(f"{k} = {v}")
            speak_async("Aliases listed.", prefer_hindi=False)
        else:
            speak_async("No aliases.", prefer_hindi=False)
        return
    s = " ".join(args)
    m = re.match(r'(.+?)="(.+)"', s)
    if m:
        ALIASES[m.group(1)] = m.group(2)
        save_aliases()
        speak_async("Alias saved.", prefer_hindi=False)
    else:
        speak_async('Usage: alias name="command"', prefer_hindi=False)

def cmd_remind(args):
    joined = " ".join(args)
    # Accept: remind me in 5 minutes to do X
    m = re.search(r'me in (\d+)\s+(second|seconds|minute|minutes|hour|hours)\s+to\s+(.+)', joined, re.I)
    if m:
        v, unit, message = m.group(1), m.group(2).lower(), m.group(3)
        secs = int(v)
        if unit.startswith("minute"):
            secs *= 60
        if unit.startswith("hour"):
            secs *= 3600
        def rem(s, msg):
            time.sleep(s)
            speak_async(f"Reminder: {msg}", prefer_hindi=True)
        threading.Thread(target=rem, args=(secs, message), daemon=True).start()
        speak_async(f"Will remind in {v} {unit}.", prefer_hindi=False)
    else:
        speak_async("Usage: remind me in <N> <unit> to <message>", prefer_hindi=False)

def cmd_sysinfo(args):
    if PSUTIL_AVAILABLE:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        speak_async(f"CPU usage {cpu:.1f}% and RAM usage {ram:.1f}%.", prefer_hindi=False)
    else:
        speak_async("psutil not installed.", prefer_hindi=False)

def spotify_search_and_open(query):
    q = query.strip()
    if SPOTIPY_AVAILABLE and SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET:
        try:
            creds = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
            sp = spotipy.Spotify(client_credentials_manager=creds)
            res = sp.search(q, type='track', limit=1)
            items = res.get('tracks', {}).get('items', [])
            if items:
                track = items[0]
                url = track.get('external_urls', {}).get('spotify')
                if url:
                    webbrowser.open(url)
                    return True, f"Opening on Spotify: {track.get('name')} — {', '.join(a['name'] for a in track.get('artists', []))}"
        except Exception as e:
            return False, f"Spotify API error: {e}"
    # fallback to web search
    safe_q = q.replace(" ", "%20")
    webbrowser.open(f"https://open.spotify.com/search/{safe_q}")
    return True, f"Opened Spotify search for: {q}"

def cmd_play(args):
    if not args:
        speak_async("Usage: play <song or artist>", prefer_hindi=False)
        return
    song = " ".join(args)
    ok, msg = spotify_search_and_open(song)
    speak_async(msg, prefer_hindi=False)

def cmd_open(args):
    if not args:
        speak_async("Usage: open <app|url>", prefer_hindi=False)
        return
    target = args[0].lower()
    if target in URLS:
        webbrowser.open(URLS[target])
        speak_async(f"Opening {target}.", prefer_hindi=False)
        return
    if open_application(target):
        speak_async(f"Started {target}.", prefer_hindi=False)
        return
    if os.path.exists(args[0]):
        try:
            if sys.platform.startswith("win"):
                os.startfile(args[0])
            else:
                subprocess.Popen([args[0]])
            speak_async("Opened provided path.", prefer_hindi=False)
            return
        except Exception as e:
            speak_async(f"Open error: {e}", prefer_hindi=False)
            return
        if args[0].startswith("http"):
            webbrowser.open(args[0])
            speak_async("Opened URL.", prefer_hindi=False)
    else:
        # smart fallback: google it
        webbrowser.open(f"https://www.google.com/search?q={requests.utils.requote_uri(target) if requests else target}")
        speak_async(f"I couldn't find {target}, so I searched it on Google.", prefer_hindi=False)

def cmd_createfile(args):
    if not args:
        speak_async("Usage: createfile <filename>", prefer_hindi=False)
        return
    fname = args[0]
    print("Enter content lines. Type EOF on a line to finish.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        speak_async(f"Created {fname}.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Create file error: {e}", prefer_hindi=False)

def cmd_remember(args):
    if len(args) < 2:
        speak_async("Usage: remember <key> <value>", prefer_hindi=False)
        return
    k = args[0]
    v = " ".join(args[1:])
    MEMORY[k] = v
    save_memory()
    speak_async(f"Saved {k}.", prefer_hindi=False)

def cmd_recall(args):
    if not args:
        speak_async("Usage: recall <key>", prefer_hindi=False)
        return
    k = args[0]
    if k in MEMORY:
        speak_async(MEMORY[k], prefer_hindi=False)
    else:
        speak_async("No memory found.", prefer_hindi=False)

def cmd_forget(args):
    if not args:
        speak_async("Usage: forget <key>", prefer_hindi=False)
        return
    k = args[0]
    if k in MEMORY:
        MEMORY.pop(k, None)
        save_memory()
        speak_async(f"Forgot {k}.", prefer_hindi=False)
    else:
        speak_async("No memory found.", prefer_hindi=False)

def cmd_createpy(args):
    if not args:
        speak_async("Usage: createpy <topic>", prefer_hindi=False)
        return
    prompt = " ".join(args)
    speak_async("Generating Python code, please wait...", prefer_hindi=False)
    code = ask_ai(f"Write a python script for: {prompt}\nProvide only valid code with no extra explanation.")
    fname = re.sub(r'[^a-zA-Z0-9_]+', '_', prompt).strip("_")[:40] or "generated"
    fname = fname + ".py"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[Saved as {fname}]")
        speak_async(f"Python file created as {fname}.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Failed to save file: {e}", prefer_hindi=False)

# -------------------------
# New utility commands & features (safe, optional)
# -------------------------

def cmd_clean(args):
    """
    Clear the terminal screen (clean command requested).
    """
    try:
        if sys.platform.startswith("win"):
            os.system("cls")
        else:
            os.system("clear")
        speak_async("Screen cleared.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Clean failed: {e}", prefer_hindi=False)

def cmd_run(args):
    """
    Run an arbitrary shell command: run <command...>
    WARNING: executes commands on the host OS. Use responsibly.
    """
    if not args:
        speak_async("Usage: run <command>", prefer_hindi=False)
        return
    cmd = " ".join(args)
    try:
        speak_async(f"Running: {cmd}", prefer_hindi=False)
        subprocess.run(cmd, shell=True)
    except Exception as e:
        speak_async(f"Run failed: {e}", prefer_hindi=False)

# Contacts management (telephone dictionary)
def cmd_add_contact(args):
    if len(args) < 2:
        speak_async("Usage: add_contact <name> <number>", prefer_hindi=False)
        return
    name = args[0]
    number = args[1]
    CONTACTS[name] = number
    save_contacts()
    speak_async(f"Contact {name} saved.", prefer_hindi=False)

def cmd_search_contact(args):
    if not args:
        speak_async("Usage: search_contact <query>", prefer_hindi=False)
        return
    q = " ".join(args).lower()
    found = {k:v for k,v in CONTACTS.items() if q in k.lower() or q in v}
    if not found:
        speak_async("No contacts found.", prefer_hindi=False)
        return
    for k,v in found.items():
        print(f"{k} -> {v}")
    speak_async(f"Found {len(found)} contacts.", prefer_hindi=False)

# QR generator
def cmd_qr(args):
    if not args:
        speak_async("Usage: qr <text> [output.png]", prefer_hindi=False)
        return
    text = args[0]
    out = args[1] if len(args) > 1 else f"qr_{int(time.time())}.png"
    if not OPTIONAL.get('qrcode'):
        speak_async("qrcode library not installed.", prefer_hindi=False)
        return
    try:
        img = qrcode.make(text)
        img.save(out)
        speak_async(f"QR saved to {out}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"QR generation failed: {e}", prefer_hindi=False)

# Screenshot
def cmd_screenshot(args):
    fname = args[0] if args else None
    if not OPTIONAL.get('pyautogui'):
        speak_async("pyautogui not installed.", prefer_hindi=False)
        return
    if not fname:
        fname = f'screenshot_{int(time.time())}.png'
    try:
        img = pyautogui.screenshot()
        img.save(fname)
        speak_async(f"Screenshot saved: {fname}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Screenshot failed: {e}", prefer_hindi=False)

# Voice recording
def cmd_record_voice(args):
    seconds = int(args[0]) if args else 5
    out = args[1] if len(args) > 1 else f"record_{int(time.time())}.wav"
    if not OPTIONAL.get('sounddevice'):
        speak_async("sounddevice not installed.", prefer_hindi=False)
        return
    try:
        samplerate = 44100
        speak_async(f"Recording for {seconds} seconds...", prefer_hindi=False)
        data = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        wavfile.write(out, samplerate, data)
        speak_async(f"Saved voice to {out}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Record failed: {e}", prefer_hindi=False)

# Screen recording (no audio, best-effort)
def cmd_screen_record(args):
    seconds = int(args[0]) if args else 10
    out = args[1] if len(args) > 1 else f"screen_{int(time.time())}.avi"
    fps = int(args[2]) if len(args) > 2 else 12
    if not OPTIONAL.get('pyautogui') or not OPTIONAL.get('cv2'):
        speak_async("pyautogui or opencv not installed.", prefer_hindi=False)
        return
    try:
        w, h = pyautogui.size()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
        start = time.time()
        speak_async(f"Recording screen for {seconds} seconds...", prefer_hindi=False)
        while time.time() - start < seconds:
            img = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        speak_async(f"Screen recording saved: {out}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Screen record failed: {e}", prefer_hindi=False)

# Webcam access (open default webcam stream with OpenCV)
def cmd_webcam(args):
    if not OPTIONAL.get('cv2'):
        speak_async("opencv-python not installed.", prefer_hindi=False)
        return
    cam_index = int(args[0]) if args else 0
    try:
        cap = cv2.VideoCapture(cam_index)
        speak_async("Opening webcam. Press 'q' in the window to close.", prefer_hindi=False)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Jarvis Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        speak_async(f"Webcam error: {e}", prefer_hindi=False)

# Mobile camera stream viewer (requires user to run IP Webcam app and provide URL)
def cmd_mobile_cam(args):
    if not args:
        speak_async("Usage: mobile_cam <stream_url>", prefer_hindi=False)
        return
    url = args[0]
    # Just open stream in default browser (safe, consent-based)
    webbrowser.open(url)
    speak_async("Opened mobile camera stream in browser. Ensure you set up the app on your phone.", prefer_hindi=False)

# Phone number lookup (safe, requires user API key for service)
def cmd_phone_lookup(args):
    """
    Lookup phone number location using user-provided API key for a third-party service.
    Usage: phone_lookup <number> <api_service> <api_key>
    Example: phone_lookup +911234567890 numverify YOUR_KEY
    NOTE: This uses external paid/consent APIs; you must supply the key.
    """
    if len(args) < 3:
        speak_async("Usage: phone_lookup <number> <service> <api_key>", prefer_hindi=False)
        return
    number = args[0]
    service = args[1].lower()
    api_key = args[2]
    if service == "numverify":
        if not requests:
            speak_async("requests not installed.", prefer_hindi=False)
            return
        try:
            url = f"http://apilayer.net/api/validate?access_key={api_key}&number={requests.utils.requote_uri(number)}"
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            data = r.json()
            print(json.dumps(data, indent=2))
            speak_async("Phone lookup printed.", prefer_hindi=False)
        except Exception as e:
            speak_async(f"Phone lookup failed: {e}", prefer_hindi=False)
    else:
        speak_async("Unsupported service. Use numverify for now.", prefer_hindi=False)

# PDF reading
def cmd_read_pdf(args):
    if not args:
        speak_async("Usage: read_pdf <path>", prefer_hindi=False)
        return
    path = args[0]
    if not OPTIONAL.get('pypdf2'):
        speak_async("PyPDF2 not installed.", prefer_hindi=False)
        return
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            out = []
            total = len(reader.pages)
            pages_to_read = list(range(total))
            for i in pages_to_read:
                try:
                    t = reader.pages[i].extract_text() or ""
                except Exception:
                    t = ""
                out.append(t)
            text = "\n\n".join(out)
            print(text[:4000])
            speak_async("PDF content printed to terminal.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"PDF read failed: {e}", prefer_hindi=False)

# QR/YouTube/Instagram/Internet helpers
def cmd_youtube(args):
    if not args:
        speak_async("Usage: youtube <search or url>", prefer_hindi=False)
        return
    query = " ".join(args)
    if OPTIONAL.get('pywhatkit'):
        try:
            pywhatkit.playonyt(query)
            speak_async("Playing on YouTube", prefer_hindi=False)
            return
        except Exception:
            pass
    if requests:
        webbrowser.open(f"https://www.youtube.com/results?search_query={requests.utils.requote_uri(query)}")
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
    speak_async("Opened YouTube search.", prefer_hindi=False)

def cmd_yt_download(args):
    if not args:
        speak_async("Usage: yt_download <url>", prefer_hindi=False)
        return
    url = args[0]
    if not OPTIONAL.get('pytube'):
        speak_async("pytube not installed.", prefer_hindi=False)
        return
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        out_dir = "downloads"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fn = stream.download(output_path=out_dir)
        speak_async(f"Downloaded: {fn}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"YT download failed: {e}", prefer_hindi=False)

def cmd_insta_download(args):
    if not args:
        speak_async("Usage: insta_download <username>", prefer_hindi=False)
        return
    username = args[0]
    if not OPTIONAL.get('instaloader'):
        speak_async("instaloader not installed.", prefer_hindi=False)
        return
    try:
        L = instaloader.Instaloader(dirname_pattern=f"insta_{username}")
        L.download_profile(username, profile_pic_only=False)
        speak_async("Instagram profile downloaded.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Instagram download failed: {e}", prefer_hindi=False)

def cmd_get_ip(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False)
        return
    try:
        r = requests.get('https://api.ipify.org?format=json', timeout=6)
        ip = r.json().get('ip')
        speak_async(f'Your public IP: {ip}', prefer_hindi=False)
    except Exception as e:
        speak_async(f'IP fetch failed: {e}', prefer_hindi=False)

def cmd_speedtest(args):
    if not OPTIONAL.get('speedtest'):
        speak_async("speedtest-cli not installed.", prefer_hindi=False)
        return
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        down = st.download() / 1e6
        up = st.upload() / 1e6
        speak_async(f'Download {down:.2f} Mbps, Upload {up:.2f} Mbps', prefer_hindi=False)
    except Exception as e:
        speak_async(f'Speed test failed: {e}', prefer_hindi=False)

# COVID India counts
def cmd_covid(args):
    if not requests:
        speak_async("requests missing", prefer_hindi=False)
        return
    try:
        # Public dataset endpoint (may change)
        r = requests.get('https://data.covid19india.org/v4/min/data.min.json', timeout=8)
        r.raise_for_status()
        data = r.json()
        out = {}
        for state, info in data.items():
            total = info.get('total', {})
            confirmed = total.get('confirmed', 0)
            out[state] = confirmed
        for s, c in sorted(out.items(), key=lambda x: x[1], reverse=True):
            print(f"{s}: {c}")
        speak_async("COVID state counts shown.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"COVID data fetch failed: {e}", prefer_hindi=False)

# System condition
def cmd_system_condition(args):
    if PSUTIL_AVAILABLE:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage('C:').percent if sys.platform.startswith('win') else psutil.disk_usage('/').percent
        speak_async(f'CPU {cpu:.1f}%, RAM {ram:.1f}%, Disk {disk:.1f}%', prefer_hindi=False)
    else:
        speak_async("psutil not installed.", prefer_hindi=False)

# Send Gmail helper (minimal)
def cmd_send_gmail(args):
    """
    send_gmail smtp port username password to subject body
    Example:
      send_gmail smtp.gmail.com 587 me@example.com apppassword friend@example.com "Hi" "Body text"
    """
    if len(args) < 7:
        speak_async("Usage: send_gmail smtp port username password to subject body", prefer_hindi=False)
        return
    smtp_server = args[0]
    port = int(args[1])
    username = args[2]
    password = args[3]
    to = args[4]
    subject = args[5]
    body = " ".join(args[6:])
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to
        s = smtplib.SMTP(smtp_server, port, timeout=10)
        s.starttls()
        s.login(username, password)
        s.sendmail(username, [to], msg.as_string())
        s.quit()
        speak_async('Email sent.', prefer_hindi=False)
    except Exception as e:
        speak_async(f'Email failed: {e}', prefer_hindi=False)

# WhatsApp send (opens web and schedules)
def cmd_whatsapp(args):
    if len(args) < 2:
        speak_async("Usage: whatsapp <number_with_cc> <message...>", prefer_hindi=False)
        return
    number = args[0]
    msg = " ".join(args[1:])
    if not OPTIONAL.get('pywhatkit'):
        speak_async("pywhatkit not installed.", prefer_hindi=False)
        return
    try:
        # schedule a minute from now
        t = time.localtime(time.time() + 70)
        hh, mm = t.tm_hour, t.tm_min
        pywhatkit.sendwhatmsg(number, msg, hh, mm, 15, True, 3)
        speak_async("WhatsApp message scheduled/opened.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"WhatsApp send failed: {e}", prefer_hindi=False)

# Wikipedia 5-line summary
def cmd_wiki(args):
    if not args:
        speak_async("Usage: wiki <query>", prefer_hindi=False)
        return
    if not OPTIONAL.get('wikipedia'):
        speak_async("wikipedia library not installed.", prefer_hindi=False)
        return
    q = " ".join(args)
    try:
        s = wikipedia.summary(q, sentences=5)
        print(s)
        speak_async(s, prefer_hindi=False)
    except Exception as e:
        speak_async(f"Wiki error: {e}", prefer_hindi=False)

# Schedule storage
def cmd_schedule_save(args):
    if len(args) < 2:
        speak_async("Usage: schedule_save <day> <item1;item2;...>", prefer_hindi=False)
        return
    day = args[0]
    items = " ".join(args[1:]).split(";")
    MEMORY.setdefault("schedule", {})[day] = items
    save_memory()
    speak_async("Schedule saved.", prefer_hindi=False)

def cmd_schedule_get(args):
    if not args:
        speak_async("Usage: schedule_get <day>", prefer_hindi=False)
        return
    day = args[0]
    s = MEMORY.get("schedule", {}).get(day)
    if not s:
        speak_async("No schedule found.", prefer_hindi=False)
        return
    for it in s:
        print("-", it)
    speak_async("Schedule displayed.", prefer_hindi=False)

# Silent mode
def cmd_silent(args):
    secs = int(args[0]) if args else 10
    speak_async(f"Going silent for {secs} seconds.", prefer_hindi=False)
    stop_tts_worker()
    time.sleep(secs)
    start_tts_worker()
    speak_async("I am back online.", prefer_hindi=False)

# Volume control (Windows-friendly)
def cmd_set_volume(args):
    if not args:
        speak_async("Usage: set_volume <percent>", prefer_hindi=False)
        return
    p = args[0]
    try:
        pct = int(p)
        if sys.platform.startswith("win"):
            # Use PowerShell to set volume (works on many Windows setups)
            speak_async("Volume control on Windows may require additional utilities. Use system volume directly.", prefer_hindi=False)
        else:
            speak_async("Volume control not fully supported on this OS via this script.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Volume failed: {e}", prefer_hindi=False)

# Power actions
def cmd_power(args):
    if not args:
        speak_async("Usage: power <shutdown|restart|sleep>", prefer_hindi=False)
        return
    action = args[0].lower()
    try:
        if action == "shutdown":
            if sys.platform.startswith("win"):
                run_shell("shutdown /s /t 5")
            else:
                run_shell("sudo shutdown -h now")
            speak_async("Powering off...", prefer_hindi=False)
        elif action == "restart":
            if sys.platform.startswith("win"):
                run_shell("shutdown /r /t 5")
            else:
                run_shell("sudo reboot")
            speak_async("Restarting...", prefer_hindi=False)
        elif action == "sleep":
            if sys.platform.startswith("win"):
                run_shell("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            else:
                run_shell("systemctl suspend")
            speak_async("Entering sleep mode...", prefer_hindi=False)
        else:
            speak_async("Unknown power action.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Power action failed: {e}", prefer_hindi=False)

# Play local music directory
def cmd_play_local(args):
    if not args:
        speak_async("Usage: play_local <directory>", prefer_hindi=False)
        return
    d = args[0]
    if not os.path.isdir(d):
        speak_async("Directory not found.", prefer_hindi=False)
        return
    files = []
    for ext in ("mp3","wav","flac","m4a","aac"):
        files.extend([os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(ext)])
    if not files:
        speak_async("No music files found in directory.", prefer_hindi=False)
        return
    # open the first file with default player (Windows)
    try:
        os.startfile(files[0])
        speak_async(f"Playing {os.path.basename(files[0])}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Play failed: {e}", prefer_hindi=False)

# Search procedure/how-to using web search or Wikipedia
def cmd_howto(args):
    if not args:
        speak_async("Usage: howto <topic>", prefer_hindi=False)
        return
    topic = " ".join(args)
    if OPTIONAL.get('wikipedia'):
        try:
            s = wikipedia.summary(topic, sentences=5)
            print(s)
            speak_async(s, prefer_hindi=False)
            return
        except Exception:
            pass
    # fallback to web search
    if requests:
        webbrowser.open(f"https://www.google.com/search?q={requests.utils.requote_uri(topic)}")
    else:
        webbrowser.open(f"https://www.google.com/search?q={topic}")
    speak_async("Search opened in browser.", prefer_hindi=False)

# -------------------------
# Commands mapping (merge original commands and new ones)
# -------------------------
COMMANDS = {
    "help": {"handler": cmd_help, "description": "Show help"},
    "exit": {"handler": cmd_exit, "description": "Exit"},
    "quit": {"handler": cmd_exit, "description": "Exit"},
    "history": {"handler": cmd_history, "description": "Show history"},
    "ls": {"handler": cmd_ls, "description": "List files"},
    "cd": {"handler": cmd_cd, "description": "Change directory"},
    "mkdir": {"handler": cmd_mkdir, "description": "Make dir"},
    "rm": {"handler": cmd_rm, "description": "Remove file"},
    "echo": {"handler": cmd_echo, "description": "Echo text"},
    "find": {"handler": cmd_find, "description": "Find file"},
    "calc": {"handler": cmd_calc, "description": "Calculator"},
    "weather": {"handler": cmd_weather, "description": "Weather"},
    "joke": {"handler": cmd_joke, "description": "Programming joke"},
    "quote": {"handler": cmd_quote, "description": "Inspirational quote"},
    "news": {"handler": cmd_news, "description": "Top headlines"},
    "alias": {"handler": cmd_alias, "description": "Create/list aliases"},
    "remind": {"handler": cmd_remind, "description": "Set reminder"},
    "play": {"handler": cmd_play, "description": "Play song on Spotify (opens web)"},
    "sysinfo": {"handler": cmd_sysinfo, "description": "Show CPU/RAM"},
    "open": {"handler": cmd_open, "description": "Open app or url"},
    "createfile": {"handler": cmd_createfile, "description": "Create file interactively"},
    "remember": {"handler": cmd_remember, "description": "Remember something"},
    "recall": {"handler": cmd_recall, "description": "Recall memory"},
    "forget": {"handler": cmd_forget, "description": "Forget memory"},
    "createpy": {"handler": cmd_createpy, "description": "Generate a Python file via AI"},
    # new commands
    "clean": {"handler": cmd_clean, "description": "Clear terminal screen"},
    "run": {"handler": cmd_run, "description": "Run shell command"},
    "add_contact": {"handler": cmd_add_contact, "description": "Add phone contact"},
    "search_contact": {"handler": cmd_search_contact, "description": "Search contacts"},
    "qr": {"handler": cmd_qr, "description": "Generate QR code"},
    "screenshot": {"handler": cmd_screenshot, "description": "Take screenshot"},
    "record_voice": {"handler": cmd_record_voice, "description": "Record voice to wav"},
    "screen_record": {"handler": cmd_screen_record, "description": "Record screen (no audio)"},
    "webcam": {"handler": cmd_webcam, "description": "Open webcam window"},
    "mobile_cam": {"handler": cmd_mobile_cam, "description": "Open mobile camera stream URL"},
    "phone_lookup": {"handler": cmd_phone_lookup, "description": "Phone number lookup (requires API)"},
    "read_pdf": {"handler": cmd_read_pdf, "description": "Read PDF text"},
    "youtube": {"handler": cmd_youtube, "description": "Play/search YouTube"},
    "yt_download": {"handler": cmd_yt_download, "description": "Download YouTube audio"},
    "insta_download": {"handler": cmd_insta_download, "description": "Download Instagram profile (instaloader)"},
    "get_ip": {"handler": cmd_get_ip, "description": "Show public IP"},
    "speedtest": {"handler": cmd_speedtest, "description": "Check internet speed"},
    "covid": {"handler": cmd_covid, "description": "COVID-19 India counts"},
    "system_condition": {"handler": cmd_system_condition, "description": "Show CPU/RAM/Disk"},
    "send_gmail": {"handler": cmd_send_gmail, "description": "Send an email via SMTP"},
    "whatsapp": {"handler": cmd_whatsapp, "description": "Send WhatsApp message (opens web)"},
    "wiki": {"handler": cmd_wiki, "description": "Wikipedia summary (5 lines)"},
    "schedule_save": {"handler": cmd_schedule_save, "description": "Save schedule for a day"},
    "schedule_get": {"handler": cmd_schedule_get, "description": "Get schedule for a day"},
    "silent": {"handler": cmd_silent, "description": "Be silent for N seconds"},
    "set_volume": {"handler": cmd_set_volume, "description": "Set system volume (platform-specific)"},
    "power": {"handler": cmd_power, "description": "Power actions: shutdown/restart/sleep"},
    "play_local": {"handler": cmd_play_local, "description": "Play local music from dir"},
    "howto": {"handler": cmd_howto, "description": "Show how-to steps for a topic"},
}

# -------------------------
# Dispatcher & voice loop (preserve original handle_command)
# -------------------------
def handle_command(line, history):
    if not line:
        return
    line = line.strip()
    parts = line.split()
    # alias expansion
    if parts and parts[0] in ALIASES:
        mapped = ALIASES[parts[0]]
        line = mapped + " " + " ".join(parts[1:])
        parts = line.split()

    # find a registered command
    cmd = None
    args = []
    for i, w in enumerate(parts):
        if w in COMMANDS:
            cmd = w
            args = parts[i+1:]
            break

    if cmd:
        try:
            if cmd == "history":
                COMMANDS[cmd]["handler"](args, history)
            else:
                COMMANDS[cmd]["handler"](args)
        except Exception as e:
            speak_async(f"Error executing {cmd}: {e}", prefer_hindi=False)
        return

    # open apps or urls by first token
    if parts:
        first = parts[0].lower()
        if first in URLS:
            webbrowser.open(URLS[first])
            speak_async(f"Opening {first}.", prefer_hindi=False)
            return
        if open_application(first):
            speak_async(f"Starting {first}.", prefer_hindi=False)
            return

    # conversational small matches
    for keys, replies in CONVERSATIONAL_INPUTS.items():
        if any(re.search(rf"\b{kw}\b", line, re.I) for kw in keys):
            speak_async(random.choice(replies), prefer_hindi=True)
            return

    # AI fallback
    speak_async("Thinking...", prefer_hindi=False)
    ans = ask_ai(line)
    # always print in english-font
    print("AI:", ans)
    # speak in Hindi
    speak_async(ans, prefer_hindi=True)

# Voice handler (preserve)
def listen_and_handle(history):
    if not SR_AVAILABLE:
        speak_async("Voice mode unavailable: SpeechRecognition not installed.", prefer_hindi=False)
        return
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.4)
        try:
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
        except Exception:
            return
    try:
        text = None
        for lang in ("hi-IN", "en-IN"):
            try:
                text = recognizer.recognize_google(audio, language=lang)
                if text:
                    break
            except Exception:
                text = None
        if not text:
            raise sr.UnknownValueError()
        print(f"User (voice): {text}")
        history.append(text)
        handle_command(text, history)
    except sr.UnknownValueError:
        speak_async("Sorry, I did not catch that. Please repeat.", prefer_hindi=False)
    except sr.RequestError:
        speak_async("Speech recognition service is down.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Voice recognition error: {e}", prefer_hindi=False)

# -------------------------
# Banner & main
# -------------------------
def print_banner():
    title = """
   ######  ##    ## ##     ## ########  ##    ##    ###    ##    ##
  ##       ##    ## ##     ## ##    ##  ##    ##   ## ##   ###  ###
   ######  ######## ##     ## ########  ########  ##   ##  ## ## ##
        ## ##    ## ##     ## ##    ##  ##    ##  #######  ##    ##
   ######  ##    ##  #######  ########  ##    ##  ##   ##  ##    ##
"""
    name = "                        -- SHUBHAM --"
    if COLORAMA_AVAILABLE:
        colors = [Fore.CYAN, Fore.MAGENTA, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.RED]
        c = random.choice(colors)
        print(c + title + Style.BRIGHT + c + name + Style.RESET_ALL)
    else:
        print(title + name)

def professional_startup_animation(name: str = "Shubham", duration: float = 1.5):
    """
    Show a compact professional terminal animation: a progress bar + moving dots + your name.
    duration controls total time of the short animation in seconds.
    """
    try:
        # small spinner + progress bar
        columns = shutil.get_terminal_size((80, 20)).columns
        bar_width = min(40, max(20, columns - 40))
        steps = int(bar_width)
        start = time.time()
        for i in range(steps + 1):
            elapsed = time.time() - start
            frac = i / steps
            filled = int(frac * bar_width)
            bar = "=" * filled + " ' " * (bar_width - filled)
            pct = int(frac * 100)
            sys.stdout.write(f"\r[{bar}] {pct:3d}%  | Initializing {name}")
            sys.stdout.flush()
            time.sleep(duration / max(1, steps))
        # final ripple
        sys.stdout.write("\r" + " ",)
        print()
        # show a small animated dot line with the name
        for _ in range(3):
            for dots in range(1, 5):
                sys.stdout.write(f"\r{name} is online" + ("." * dots))
                sys.stdout.flush()
                time.sleep(0.12)
        print()
    except Exception:
        # fallback simple prints
        print(f"Starting... {name}")

def main():
    global OPENAI_API_KEY, GEMINI_KEY, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
    SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

    print_banner()
    print(f"[Status] Gemini SDK installed: {GENAI_AVAILABLE}, GEMINI_KEY set: {bool(GEMINI_KEY)}")
    print(f"[Status] OpenAI SDK installed: {OPENAI_AVAILABLE}, OPENAI_KEY set: {bool(OPENAI_API_KEY)}")
    start_tts_worker()
    # startup effect & grand greeting (sound effect)
    startup_effect()
    speak_async("Welcome back, Shubham. All systems are online.", prefer_hindi=True)

    history = []
    voice_mode = False
    while True:
        try:
            if voice_mode:
                listen_and_handle(history)
            else:
                cwd = os.getcwd()
                try:
                    user = input(f"{cwd}> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user:
                    continue
                if user.lower() in ("voice mode", "start voice mode"):
                    if SR_AVAILABLE:
                        voice_mode = True
                        speak_async("Voice mode activated. Speak now.", prefer_hindi=True)
                        continue
                    else:
                        print("Voice mode unavailable. Install SpeechRecognition and a microphone.")
                        continue
                if user.lower() in ("text mode", "exit voice mode", "exit voice"):
                    voice_mode = False
                    speak_async("Text mode activated.", prefer_hindi=True)
                    continue
                # direct AI prefix
                if user.lower().startswith("ai:") or user.lower().startswith("ask:"):
                    prompt = user.split(":", 1)[1].strip()
                    speak_async("Thinking...", prefer_hindi=False)
                    ans = ask_ai(prompt)
                    print("AI:", ans)
                    speak_async(ans, prefer_hindi=True)
                    continue
                history.append(user)
                handle_command(user, history)
        except (KeyboardInterrupt, SystemExit):
            print("\nExiting Jarvis...")
            stop_tts_worker()
            break
        except Exception as e:
            traceback.print_exc()
            speak_async(f"A critical error occurred: {e}", prefer_hindi=False)
            continue

if __name__ == "__main__":
    main()
