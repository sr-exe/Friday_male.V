#!/usr/bin/env python3
"""
ultimate_jarvis_final.py
Single-file Jarvis for Windows 11 — prints in English, speaks in Hindi,
startup sound effect, responsive voice mode (manual toggle), Spotify search,
AI fallbacks, aliases & small memory storage.

Save as ultimate_jarvis_final.py and run with Python 3.8+.
"""

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

# Optional libs (guarded)
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

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except Exception:
    EDGE_TTS_AVAILABLE = False

# playsound: prefer 'playsound' (cross-platform) if available
try:
    import playsound
    PLAYSOUND_AVAILABLE = True
except Exception:
    PLAYSOUND_AVAILABLE = False

# pyttsx3 fallback
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
    _pytt_engine = pyttsx3.init()
except Exception:
    PYTTSX3_AVAILABLE = False
    _pytt_engine = None

# SpeechRecognition (voice capture)
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# spotipy (spotify api)
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except Exception:
    SPOTIPY_AVAILABLE = False

# openai (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# deep translator to translate english->hindi for TTS (optional)
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except Exception:
    DEEP_TRANSLATOR_AVAILABLE = False

# psutil for sysinfo
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# google generative ai (Gemini)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# colorama for banner colors
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False

# requests for web queries
try:
    import requests
except Exception:
    requests = None

# -------------------------
# Persistence & config
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
ALIASES_FILE = BASE_DIR / "jarvis_aliases.json"
MEMORY_FILE = BASE_DIR / "jarvis_memory.json"

def _load_json(path, default):
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

ALIASES = _load_json(ALIASES_FILE, {})
MEMORY = _load_json(MEMORY_FILE, {})

def _save_json(path, data):
    try:
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[Warning] could not save {path}: {e}")

def save_aliases():
    _save_json(ALIASES_FILE, ALIASES)

def save_memory():
    _save_json(MEMORY_FILE, MEMORY)

# Keys from environment (you may also set them in code below)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Spotify credentials (can set here or via env variables)
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")  # or set your client id
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback")

# -------------------------
# TTS & translate behavior
# -------------------------
DEFAULT_HI_VOICE = "hi-IN-MadhurNeural"
DEFAULT_EN_VOICE = "en-US-GuyNeural"

tts_queue = queue.Queue()
tts_thread = None
tts_running = threading.Event()

def _translate_to_hindi(text):
    """Translate English to Hindi if translator available."""
    if not text:
        return text
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return text
    if DEEP_TRANSLATOR_AVAILABLE:
        try:
            return GoogleTranslator(source='auto', target='hi').translate(text)
        except Exception:
            return text
    return text  # fallback: return original (edge_tts may be okay speaking English voice)

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

def speak_async(text: str, prefer_hindi: bool = True, also_print: bool = True):
    """
    Backwards-compatible:
     - always print english-font display (translate to english only if original was Devanagari)
     - speak in Hindi voice where possible (translate English->Hindi for TTS if translator available)
    """
    # prepare display text: if text contains Devanagari, translate to English for display
    display_text = text
    try:
        if any("\u0900" <= ch <= "\u097F" for ch in text) and DEEP_TRANSLATOR_AVAILABLE:
            # translate to english for printed display
            from deep_translator import GoogleTranslator as _GT
            try:
                display_text = _GT(source='auto', target='en').translate(text)
            except Exception:
                display_text = text
    except Exception:
        display_text = text

    if also_print:
        print(f"Jarvis: {display_text}")

    # prepare speak text (Hindi voice): if text already Devanagari, speak as is; else translate if prefer_hindi
    speak_text = text
    if not any("\u0900" <= ch <= "\u097F" for ch in text) and prefer_hindi:
        speak_text = _translate_to_hindi(text)

    start_tts_worker()
    tts_queue.put((speak_text, True))  # True => use Hindi voice where possible

# startup sound/effect
def startup_effect():
    try:
        # small ascii typewriter animation + beep if possible
        for s in ["[ Booting Jarvis AI OS ... ]", "[ Initializing modules ... ]", "[ Welcome, Shubham. All systems online. ]"]:
            for ch in s:
                sys.stdout.write(ch)
                sys.stdout.flush()
                time.sleep(0.005)
            print()
            time.sleep(0.12)
        # play a short beep/wav if playsound available and ship bundled tone
        tone_played = False
        if PLAYSOUND_AVAILABLE:
            # Try to synthesize a short beep using pyttsx3 or use winsound on Windows
            try:
                if sys.platform.startswith("win"):
                    import winsound
                    winsound.MessageBeep()
                    tone_played = True
                else:
                    # no guaranteed system beep on other platforms
                    pass
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# AI integration (Gemini/OpenAI) — robust attempts
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
    # attempt a small set of likely models (accounts vary)
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
            # try a few call styles
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
    # try Gemini then OpenAI then Wikipedia quick summary
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
    # fallback to wikipedia
    try:
        if requests:
            q = prompt.strip()
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={requests.utils.requote_uri(q)}&limit=1&format=json"
            r = requests.get(search_url, timeout=6)
            r.raise_for_status()
            data = r.json()
            if len(data) >= 3 and data[2]:
                snippet = data[2][0]
                url = data[3][0] if len(data) > 3 and data[3] else None
                return f"{snippet} ({url})" if url else snippet
    except Exception:
        pass
    diag = " | ".join(p for p in [("Gemini: "+str(err) if err else ""), ("OpenAI: "+str(err2) if err2 else "")] if p)
    return f"I couldn't reach cloud AI. ({diag})"

# -------------------------
# Utilities & command handlers
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
    "vscode": r"C:\Users\shubham\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "chrome": "chrome",
    "firefox": "firefox",
    "notepad": "notepad.exe",
}

URLS = {
    "whatsapp": "https://web.whatsapp.com/",
    "chatgpt": "https://chat.openai.com/",
    "git": "https://github.com/",
    "email": "https://mail.google.com/",
}

CONVERSATIONAL_INPUTS = {
    ("hello", "hi", "hey"): ["Hello there! How can I help?", "Hi! What can I do for you?", "Greetings! Ready for your command."],
    ("how are you", "how's it going", "kaise ho"): ["I'm running smoothly — thanks for asking!", "All systems nominal.", "Feeling helpful today."]
}

def cmd_help(args):
    lines = ["\n--- Jarvis Commands ---"]
    for k,v in sorted(COMMANDS.items()):
        lines.append(f"{k:<12} - {v.get('description','')}")
    print("\n".join(lines))
    speak_async("Commands displayed in terminal.", prefer_hindi=False)

def cmd_exit(args):
    speak_async("Goodbye. Signing off.", prefer_hindi=True)
    time.sleep(0.2)
    stop_tts_worker()
    sys.exit(0)

def cmd_history(args, history):
    speak_async("Showing command history.", prefer_hindi=False)
    for i,c in enumerate(history,1):
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
        speak_async("Usage: mkdir <dir>", prefer_hindi=False); return
    try:
        os.makedirs(args[0], exist_ok=True)
        speak_async("Directory created.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"mkdir error: {e}", prefer_hindi=False)

def cmd_rm(args):
    if not args:
        speak_async("Usage: rm <file>", prefer_hindi=False); return
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
        speak_async("Usage: find <filename>", prefer_hindi=False); return
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
        speak_async("Usage: calc <expression>", prefer_hindi=False); return
    expr = " ".join(args)
    try:
        res = eval(expr, {"__builtins__": {}})
        speak_async(f"Result: {res}", prefer_hindi=False)
    except Exception as e:
        speak_async(f"Calc error: {e}", prefer_hindi=False)

def cmd_weather(args):
    if not args:
        speak_async("Usage: weather <city>", prefer_hindi=False); return
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False); return
    city = " ".join(args)
    try:
        r = requests.get(f"https://wttr.in/{requests.utils.requote_uri(city)}?format=3", timeout=6)
        speak_async(r.text, prefer_hindi=False)
    except Exception as e:
        speak_async(f"Weather error: {e}", prefer_hindi=False)

def cmd_joke(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False); return
    try:
        r = requests.get("https://v2.jokeapi.dev/joke/Any?type=single&category=Programming", timeout=6)
        r.raise_for_status()
        j = r.json()
        speak_async(j.get("joke","Couldn't fetch a joke."), prefer_hindi=False)
    except Exception as e:
        speak_async(f"Joke error: {e}", prefer_hindi=False)

def cmd_quote(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False); return
    try:
        r = requests.get("https://api.quotable.io/random", timeout=6)
        r.raise_for_status()
        d = r.json()
        speak_async(f'"{d.get("content")}" - {d.get("author") or "Unknown"}', prefer_hindi=False)
    except Exception as e:
        speak_async("Could not fetch quote.", prefer_hindi=False)

def cmd_news(args):
    if not requests:
        speak_async("requests module missing.", prefer_hindi=False); return
    try:
        r = requests.get("http://feeds.bbci.co.uk/news/rss.xml", timeout=6)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall('.//item')[:5]
        for i,it in enumerate(items,1):
            t = it.find('title').text
            print(f"{i}. {t}")
        speak_async("Headlines shown.", prefer_hindi=False)
    except Exception as e:
        speak_async(f"News error: {e}", prefer_hindi=False)

def cmd_alias(args):
    if not args:
        if ALIASES:
            for k,v in ALIASES.items():
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
    m = re.search(r'me in (\d+)\s+(second|seconds|minute|minutes|hour|hours)\s+to\s+(.+)', joined, re.I)
    if m:
        v, unit, message = m.group(1), m.group(2), m.group(3)
        secs = int(v)
        if unit.startswith("minute"): secs *= 60
        if unit.startswith("hour"): secs *= 3600
        def rem(s,msg):
            time.sleep(s)
            speak_async(f"Reminder: {msg}", prefer_hindi=True)
        threading.Thread(target=rem, args=(secs,message), daemon=True).start()
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
            creds = SpotifyClientCredentials(
    client_id="8043b61cfdb34e25bd40a884c995685b",
    client_secret="d50e28eb9d3243058d6c9f14105f02e4"
)
            sp = spotipy.Spotify(client_credentials_manager=creds)
            res = sp.search(q, type='track', limit=1)
            items = res.get('tracks', {}).get('items', [])
            if items:
                track = items[0]
                url = track.get('external_urls', {}).get('spotify')
                if url:
                    webbrowser.open(url)
                    return True, f"Opening on Spotify: {track.get('name')} — {', '.join(a['name'] for a in track.get('artists',[]))}"
        except Exception as e:
            return False, f"Spotify API error: {e}"
    # fallback to web search
    safe_q = q.replace(" ", "%20")
    webbrowser.open(f"https://open.spotify.com/search/{safe_q}")
    return True, f"Opened Spotify search for: {q}"

def cmd_play(args):
    if not args:
        speak_async("Usage: play <song or artist>", prefer_hindi=False); return
    song = " ".join(args)
    ok,msg = spotify_search_and_open(song)
    speak_async(msg, prefer_hindi=False)

def cmd_open(args):
    if not args:
        speak_async("Usage: open <app|url>", prefer_hindi=False); return
    target = args[0].lower()
    if target in URLS:
        webbrowser.open(URLS[target]); speak_async(f"Opening {target}.", prefer_hindi=False); return
    if open_application(target):
        speak_async(f"Started {target}.", prefer_hindi=False); return
    if os.path.exists(args[0]):
        try:
            if sys.platform.startswith("win"):
                os.startfile(args[0])
            else:
                subprocess.Popen([args[0]])
            speak_async("Opened provided path.", prefer_hindi=False); return
        except Exception as e:
            speak_async(f"Open error: {e}", prefer_hindi=False); return
    if args[0].startswith("http"):
        webbrowser.open(args[0]); speak_async("Opened URL.", prefer_hindi=False)
    else:
        speak_async(f"Cannot find application {target}.", prefer_hindi=False)

def cmd_createfile(args):
    if not args:
        speak_async("Usage: createfile <filename>", prefer_hindi=False); return
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
        speak_async("Usage: remember <key> <value>", prefer_hindi=False); return
    k = args[0]; v = " ".join(args[1:])
    MEMORY[k] = v; save_memory()
    speak_async(f"Saved {k}.", prefer_hindi=False)

def cmd_recall(args):
    if not args:
        speak_async("Usage: recall <key>", prefer_hindi=False); return
    k = args[0]
    if k in MEMORY:
        speak_async(MEMORY[k], prefer_hindi=False)
    else:
        speak_async("No memory found.", prefer_hindi=False)

def cmd_forget(args):
    if not args:
        speak_async("Usage: forget <key>", prefer_hindi=False); return
    k = args[0]
    if k in MEMORY:
        MEMORY.pop(k, None); save_memory()
        speak_async(f"Forgot {k}.", prefer_hindi=False)
    else:
        speak_async("No memory found.", prefer_hindi=False)

def cmd_createpy(args):
    if not args:
        speak_async("Usage: createpy <topic>", prefer_hindi=False); return
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

# Commands mapping
COMMANDS = {
    "help": {"handler": cmd_help, "description":"Show help"},
    "exit": {"handler": cmd_exit, "description":"Exit"},
    "quit": {"handler": cmd_exit, "description":"Exit"},
    "history": {"handler": cmd_history, "description":"Show history"},
    "ls": {"handler": cmd_ls, "description":"List files"},
    "cd": {"handler": cmd_cd, "description":"Change directory"},
    "mkdir": {"handler": cmd_mkdir, "description":"Make dir"},
    "rm": {"handler": cmd_rm, "description":"Remove file"},
    "echo": {"handler": cmd_echo, "description":"Echo text"},
    "find": {"handler": cmd_find, "description":"Find file"},
    "calc": {"handler": cmd_calc, "description":"Calculator"},
    "weather": {"handler": cmd_weather, "description":"Weather"},
    "joke": {"handler": cmd_joke, "description":"Programming joke"},
    "quote": {"handler": cmd_quote, "description":"Inspirational quote"},
    "news": {"handler": cmd_news, "description":"Top headlines"},
    "alias": {"handler": cmd_alias, "description":"Create/list aliases"},
    "remind": {"handler": cmd_remind, "description":"Set reminder"},
    "play": {"handler": cmd_play, "description":"Play song on Spotify (opens web)"},
    "sysinfo": {"handler": cmd_sysinfo, "description":"Show CPU/RAM"},
    "open": {"handler": cmd_open, "description":"Open app or url"},
    "createfile": {"handler": cmd_createfile, "description":"Create file interactively"},
    "remember": {"handler": cmd_remember, "description":"Remember something"},
    "recall": {"handler": cmd_recall, "description":"Recall memory"},
    "forget": {"handler": cmd_forget, "description":"Forget memory"},
    "createpy": {"handler": cmd_createpy, "description":"Generate a Python file via AI"},
}

# Dispatcher & voice loop
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
    cmd = None; args = []
    for i,w in enumerate(parts):
        if w in COMMANDS:
            cmd = w; args = parts[i+1:]; break

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
            webbrowser.open(URLS[first]); speak_async(f"Opening {first}.", prefer_hindi=False); return
        if open_application(first):
            speak_async(f"Starting {first}.", prefer_hindi=False); return

    # conversational small matches
    for keys,replies in CONVERSATIONAL_INPUTS.items():
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
        for lang in ("hi-IN","en-IN"):
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

# Banner & main
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

def main():
    global OPENAI_API_KEY, GEMINI_KEY, SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
    OPENAI_API_KEY = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    GEMINI_KEY = GEMINI_KEY or os.getenv("GEMINI_API_KEY", "")
    # spotify creds may be set in env
    SPOTIPY_CLIENT_ID = SPOTIPY_CLIENT_ID or os.getenv("SPOTIPY_CLIENT_ID","")
    SPOTIPY_CLIENT_SECRET = SPOTIPY_CLIENT_SECRET or os.getenv("SPOTIPY_CLIENT_SECRET","")
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
                    prompt = user.split(":",1)[1].strip()
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
            speak_async(f"A critical error occurred: {e}", prefer_hindi=False)
            continue

if __name__ == "__main__":
    main()

