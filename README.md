# Friday_male.V
# 🤖 FRIDAY — Your Personal AI automation
### Developed by **Shubham Rathod**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-lightgrey)
![Voice%20Enabled](https://img.shields.io/badge/Voice%20Enabled-Yes-brightgreen)
![AI%20Integration](https://img.shields.io/badge/AI-Gemini%20%7C%20OpenAI%20%7C%20Local-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🌟 Overview

**FRIDAY** (Fully Responsive Intelligent Dynamic Assistant by You) is a **next-generation personal AI assistant** built completely by **Shubham Rathod** in Python.

It works both through **voice** and **text commands**, understands **Hindi & English**, speaks naturally using **text-to-speech**, and can perform intelligent actions such as opening apps, searching music, generating code, retrieving information using AI, and more — all in one file.

> 💬 A modern, powerful assistant designed for everyday tasks — locally, privately, and efficiently.

---

## 🚀 Key Features

✅ **Dual Mode Operation** — Use via *voice* or *text*  
✅ **Hindi + English Speech** — Speaks and understands both  
✅ **Fast & Natural Speech** — 1.5x rate optimized Hindi TTS  
✅ **Real-Time Command Handling** — Open apps, run calculations, check weather, play songs, and more  
✅ **AI Integrated** — Connects to Gemini or OpenAI if available  
✅ **Offline Safe Mode** — Works without internet for most commands  
✅ **Voice Mode Toggle** — Go completely hands-free  
✅ **Spotify Search** — “Play [song name]” instantly opens in Spotify  
✅ **Smart Memory** — Remember and recall information anytime  
✅ **System Info Monitor** — Quick CPU and RAM report  
✅ **AI Code Generator** — Create working Python scripts using one command  
✅ **Beautiful Console UI** — Animated startup, colors, and effects  

---

## 🧠 Capabilities

| Category | Examples |
|-----------|-----------|
| 🎤 **Voice Commands** | “Voice mode”, “Play Kesariya”, “What’s the weather in Delhi?” |
| ⚙️ **System Commands** | `ls`, `cd Desktop`, `mkdir test`, `sysinfo` |
| 🧾 **AI Interaction** | `ai: who invented AI?` or `ask: explain machine learning` |
| 💾 **Memory Commands** | `remember birthday 16 May`, `recall birthday`, `forget birthday` |
| 🎵 **Music Control** | `play Shape of You` (opens in Spotify) |
| 🧮 **Math & Logic** | `calc 15*7 + 30` |
| 📰 **News & Quotes** | `news`, `quote`, `joke` |
| 🧰 **Developer Tools** | `createpy simple game` → Generates Python code automatically |

---

## 🧩 Technology Stack

| Module | Description |
|--------|-------------|
| **Python 3.8+** | Core runtime |
| **pyttsx3** | Local text-to-speech engine |
| **edge-tts** | Neural voice synthesis (optional) |
| **SpeechRecognition** | Voice input & mic integration |
| **spotipy** | Spotify API integration |
| **deep-translator** | Automatic English ↔ Hindi translation |
| **google-generativeai / openai** | Cloud AI models for question answering |
| **psutil** | System performance monitoring |
| **colorama** | Rich terminal output |
| **requests** | Web API requests & Wikipedia fallback |

---

## 🛠️ Installation Guide

### 1️⃣ Clone or Download Repository
```bash
github clone : https://github.com/sr-exe/Friday_male.V.git
cd Friday_male.V
```
2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
If you prefer only offline functionality:

pip install pyttsx3 SpeechRecognition playsound requests colorama psutil deep-translator
```

3️⃣ Set API Keys (Optional)
``` bash
To use Gemini, OpenAI, or Spotify features, set the following environment variables:
Windows (PowerShell):

setx OPENAI_API_KEY "your_openai_key_here"
setx GEMINI_API_KEY "your_gemini_key_here"
setx SPOTIPY_CLIENT_ID "your_spotify_client_id"
setx SPOTIPY_CLIENT_SECRET "your_spotify_client_secret"
You can skip this step for offline mode.
```

4️⃣ Run FRIDAY
``` bash
python friday.py
You’ll see:

css
Copy code
[ Booting Jarvis AI OS ... ]
[ Initializing modules ... ]
[ Welcome, Shubham. All systems online. ]
Then FRIDAY will greet you in Hindi and start listening or accepting typed commands.
```

### 🧭 Usage Examples
-Action	Command
-Open VS Code	open vscode
-Search Song	play pasoori
-System Info	sysinfo
-Create Folder	mkdir newproject
-Calculate	calc (23 + 45)/2
-Ask AI	ai: explain black holes
-Voice Mode	Type voice mode and start speaking
-Text Mode	Say or type text mode to switch back

### 🧰 Advanced Capabilities
***🧩 AI Auto-Fallback:***
-If Gemini or OpenAI keys aren’t available, FRIDAY fetches information via Wikipedia automatically.

### 🎙️ Voice Recognition:
-Listens actively, interprets Hindi or English voice commands, and performs related actions.

### ⚡ Optimized TTS Engine:
-FRIDAY speaks at 1.5x faster rate for more human-like rhythm.

### 💾 Persistent Memory:
-Stores your reminders or facts inside small local .json files — no data ever leaves your system.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 🔐 Privacy & Safety
-✅ No cloud upload unless using AI queries.
-✅ All system control and voice recognition are processed locally.
-✅ API keys are optional and stored safely in your environment variables.
-✅ No hidden logging or analytics.
-✅ 100% offline-compatible design.

### 🧑‍💻 Project Structure
``` bash

Friday_male.V
│
├── friday.py                # Main executable file
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
```

<img width="1311" height="871" alt="Screenshot 2025-10-05 183638" src="https://github.com/user-attachments/assets/51948c51-c74e-466d-a2b0-6d614b60747b" />

### 👨‍💻 About the Creator
----- Shubham Rathod ----- 
---
Aspiring new technologies | Stuudent | Automation Enthusiast

### 💬 “I built FRIDAY to make daily computing smarter, faster, and more natural — using my own AI-powered assistant.”

---

📧Linkedin : [😍Shubham](https://www.linkedin.com/in/shubham-rathod-337b40384/)
### ⭐ If you like this project, don’t forget to star the repository!

---

### ⚠️ Note
This project is fully developed and owned by Shubham Rathod.
You are free to view and use the source code for learning, but redistribution or commercial use requires the author’s permission.

---

### 🗣️ “Your voice, your AI — meet FRIDAY.

— Developed with 💙 by Shubham Rathod


---

© 2025 All Rights Reserved : Shubham Rathod
