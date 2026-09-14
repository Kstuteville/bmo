# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import random
import re
import sys
import select
import traceback
import atexit
import datetime
import warnings
import wave
import struct 
import queue
import math

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal 

# --- AI ENGINES ---
import openwakeword
from openwakeword.model import Model
import ollama 

# --- WEB SEARCH (Using your working import) ---
from duckduckgo_search import DDGS 

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
# secrets.json is gitignored and merged OVER config.json. The calendar iCal URL is a
# password -- anyone holding it can read the calendar -- and config.json is tracked.
SECRETS_FILE = "secrets.json"
MEMORY_FILE = "memory.json"
# Bump this whenever the persona or memory shape changes. Older files are discarded
# on load: stale turns from a previous persona poison context -- Gemma-era history
# is why BMO kept answering "Sparky" and "November 2023" after the rewrite.
MEMORY_VERSION = 2
# How many past messages (not exchanges) to send. Each exchange is 2. Small models pay
# for every token of prompt, so this is the main latency dial that is not the model itself.
MEMORY_TURNS = 8   # overridden by config below
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "vision_model": "moondream",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    "calendar_ics_url": "",          # put the real one in secrets.json, not here
    "reminder_lead_minutes": 30,
    "calendar_poll_minutes": 5
}

# LLM SETTINGS
OLLAMA_OPTIONS = {
    'keep_alive': '-1',     
    'num_thread': 4,
    'temperature': 0.7,     
    'top_k': 40,
    'top_p': 0.9,
    # Hard cap on reply length. BMO speaks out loud, so a long answer is both
    # out of character and the main latency cost on a Pi CPU.
    'num_predict': 120
}

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")

    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r") as f:
                secrets = json.load(f)
            config.update({k: v for k, v in secrets.items() if not k.startswith("_")})
            print(f"[CFG] merged {SECRETS_FILE}", flush=True)
        except Exception as e:
            print(f"[CFG] {SECRETS_FILE} unreadable: {e}", flush=True)
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
MEMORY_TURNS = int(CURRENT_CONFIG.get("memory_turns", MEMORY_TURNS))
VISION_MODEL = CURRENT_CONFIG["vision_model"]

def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        print(f"[AUDIO] Input device index not found: {index}", flush=True)
        return None

    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        print(f"[AUDIO DEBUG] Index {idx}: {dev.get('name')} (In: {dev.get('max_input_channels')})", flush=True) # DEBUG LINE
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx

    print(f"[AUDIO] Input device name not found: {requested}", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        print(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}", flush=True)
    except Exception:
        print(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}", flush=True)

def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        device_info = sd.query_devices(device)
        print(f"[AUDIO DEBUG] Device Info: {device_info}", flush=True) # DEBUG
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception as e:
        print(f"[AUDIO DEBUG] Query failed: {e}", flush=True)
        pass

    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue

    return int(candidates[0]) if candidates else 44100

class BotStates:
    IDLE = "idle"             
    LISTENING = "listening"   
    THINKING = "thinking"     
    SPEAKING = "speaking"     
    ERROR = "error"           
    CAPTURING = "capturing" 
    WARMUP = "warmup"       

# --- PERSONA + LIVE STATE ---
# The persona is rebuilt on EVERY turn and never persisted to memory.json.
# Persisting it is what let an old system prompt outlive config.json (BMO called
# himself "Sparky" because the saved memory still carried the previous prompt).
BASE_PERSONA = """You are BMO, a small teal robot companion. Cheerful, playful, a little childlike.

Rules:
- You speak OUT LOUD. One or two short sentences, always.
- Answer directly. Never narrate your actions or mention your tools.
- You know the date and time already. Never guess them, never use a tool for them.
- Use a tool ONLY when you truly need outside info or your camera. Things you already
  know (maths, facts, chat) need no tool.
- But when you DO need one, just use it. Never ask permission, never say you lack
  real-time access.
- Anything current (news, weather, showtimes, prices, scores, hours) needs search_web.
- The user's schedule, calendar, meetings, appointments or classes need get_agenda.
- NEVER invent calendar events or describe what you see. Use the tool or say you cannot."""

def build_system_prompt(mood_line=""):
    """Persona + live state, regenerated per turn.

    Date/time is INJECTED, not retrieved and not a tool: the OS already knows it
    exactly, so a tool call would be slower and could still be wrong. Without this
    the model answers from training data (it once claimed November 2023).
    """
    now = datetime.datetime.now()
    stamp = (f"{now.strftime('%A, %B %d, %Y')}, "
             f"{now.hour % 12 or 12}:{now.minute:02d} "
             f"{'AM' if now.hour < 12 else 'PM'}")
    parts = [BASE_PERSONA, f"Current date and time: {stamp}."]

    # Who BMO is talking to. Same reasoning as the date: stable, known in advance,
    # so inject it rather than making BMO retrieve or be told it every session.
    # Things BMO LEARNS about you later belong in the Phase 3 facts store instead.
    user = CURRENT_CONFIG.get("user_name", "").strip()
    if user:
        parts.append(f"You are talking to {user}. Use their name occasionally, not every reply.")
    extras = CURRENT_CONFIG.get("system_prompt_extras", "").strip()
    if extras:
        parts.append(extras)
    if mood_line:
        parts.append(mood_line)          # Phase 2 hook; empty for now
    return "\n\n".join(parts)

# Calendar questions are pre-routed past the model entirely. qwen2.5:1.5b will not
# reliably emit a tool_call for them -- it invents events instead ("special meeting with
# friends"). Matching here means the model never gets to decide, so it cannot invent.
# It is also FASTER: skips the tool-decision pass, so one model call instead of two.
CALENDAR_RE = re.compile(
    r"\b(calendar|schedule|agenda|meeting|meetings|appointment|appointments"
    r"|class|classes|plans)\b", re.I)

# --- TOOL SCHEMAS (Ollama native function calling) ---
# The runtime validates these, so the model cannot invent a tool name or a
# parameter key. This replaces regex-scraping JSON out of prose.
# NOTE: there is deliberately no get_time tool -- see build_system_prompt().
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": ("Search the web for anything current or that you do not "
                            "know: news, weather, movie showtimes, prices, sports "
                            "scores, opening hours, recent events. Call this "
                            "immediately; never ask the user for permission first."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agenda",
            "description": ("Read the user's real calendar. Use for ANY question about "
                            "their schedule, agenda, meetings, appointments, classes, "
                            "plans, or what they have on today or tomorrow."),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "'today' or 'tomorrow'"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": ("Take a photo with your camera and look at what is in front "
                            "of you. Use when asked what you can see."),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"

# =========================================================================
# CALENDAR (read-only, via Google's secret iCal URL -- no OAuth)
# =========================================================================
# Google's push notifications are webhooks needing a public HTTPS endpoint, which a
# Pi behind a home router does not have. So we poll the .ics feed and fire our own
# reminders from each event's start time. Note the feed is cached by Google and can
# lag a few minutes behind live edits.

_CAL_CACHE = {"fetched_at": None, "events": []}

def _local_tz():
    return datetime.datetime.now().astimezone().tzinfo

def fetch_events(force=False):
    """Return today's and tomorrow's events as [{uid, summary, start}] in local time.

    Soft-fails to [] on every error: the calendar is an enhancement, and BMO must keep
    working as a normal assistant with no URL, no network, or a malformed feed.
    """
    url = (CURRENT_CONFIG.get("calendar_ics_url") or "").strip()
    if not url:
        return []

    poll_min = int(CURRENT_CONFIG.get("calendar_poll_minutes", 5))
    now = datetime.datetime.now(tz=_local_tz())
    if (not force and _CAL_CACHE["fetched_at"]
            and (now - _CAL_CACHE["fetched_at"]).total_seconds() < poll_min * 60):
        return _CAL_CACHE["events"]

    try:
        import icalendar
        import recurring_ical_events
        import urllib.request

        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read()
        cal = icalendar.Calendar.from_ical(raw)

        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + datetime.timedelta(days=2)
        # Expands RRULE into real occurrences -- without this, recurring meetings
        # (standups, weeklies) would silently never produce a reminder.
        occurrences = recurring_ical_events.of(cal).between(start, end)

        events = []
        for ev in occurrences:
            dt = ev.get("DTSTART").dt
            if not isinstance(dt, datetime.datetime):       # all-day event
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_local_tz())
            events.append({
                "uid": f"{ev.get('UID', '')}-{dt.isoformat()}",
                "summary": str(ev.get("SUMMARY", "something")),
                "start": dt.astimezone(_local_tz()),
            })
        events.sort(key=lambda e: e["start"])

        _CAL_CACHE["fetched_at"] = now
        _CAL_CACHE["events"] = events
        print(f"[CAL] fetched {len(events)} events", flush=True)
        return events

    except ImportError as e:
        print(f"[CAL] missing dependency ({e}); run ./setup.sh", flush=True)
    except Exception as e:
        print(f"[CAL] fetch failed: {e}", flush=True)
    return []

def events_for_day(day="today"):
    now = datetime.datetime.now(tz=_local_tz())
    target = (now + datetime.timedelta(days=1)).date() if str(day).lower().startswith("tomorrow") else now.date()
    return [e for e in fetch_events() if e["start"].date() == target]

def format_agenda(events, day="today", who=""):
    """Build the spoken calendar answer with no model involved.

    Step 4 (LLM phrasing) is removed for calendar deliberately: it was the only
    stochastic link in an otherwise exact chain, and the one place a wrong answer
    actually costs the user something. This is 100% accurate and instant.
    """
    lead = f"Okay {who}! " if who else "Okay! "
    if not events:
        return f"{lead}Nothing on your calendar {day}."
    parts = [f"{e['summary']} at {describe_time(e['start'])}." for e in events]
    n = len(parts)
    return f"{lead}You have {n} thing{'s' if n != 1 else ''} {day}. " + " Then ".join(parts)

def describe_time(dt):
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480 
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300 

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        master.attributes('-fullscreen', True) 
        master.bind('<Escape>', self.exit_fullscreen)
        
        # Inputs
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)
        
        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0 
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        
        self.last_ptt_time = 0 
        self.ptt_event = threading.Event()       
        self.interject_event = threading.Event()   # set by the reminder ticker
        self.announced = set()                     # event uids already announced
        self.pending_interjection = None
        self.recording_active = threading.Event() 
        self.interrupted = threading.Event() 
        
        self.tts_queue = []          
        self.tts_queue_lock = threading.Lock() 
        self.tts_thread = None       
        self.tts_active = threading.Event()
        self.current_audio_process = None 
        self.exiting = False
        
        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                print("[INIT] Wake Word Loaded.", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (New API).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Failed to load model: {e}")
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        else:
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility) 
        
        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)
        
        self.response_text = tk.Text(master, height=6, width=60, wrap=tk.WORD, 
                                     state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12)) 
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation() 
        self.master.after(15_000, self.check_reminders)
        
        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except: pass

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear() 
        
        self.save_chat_history()
        
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except: pass
        try:
            sd.stop()
        except: pass

        try:
            self.master.quit()
        except Exception:
            pass
        
    def exit_fullscreen(self, event=None):
        self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError: pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5: 
            return 
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear() 
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set() 
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state == BotStates.SPEAKING or self.current_state == BotStates.THINKING:
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try: self.current_audio_process.terminate()
                except: pass
            self.set_state(BotStates.IDLE, "Interrupted.")

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"] 
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if state in self.animations.get("idle", []):
                     self.animations[state] = self.animations["idle"]
                else:
                    # Blue screen fallback
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) > 1:
                self.current_frame_index = random.randint(1, len(frames) - 1)
            else:
                self.current_frame_index = 0 
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg: print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except: pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            if newline: 
                self.response_text.insert(tk.END, text + "\n")
            else: 
                self.response_text.insert(tk.END, text)
            
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
            
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END) 
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, update_text_stream)

    # =========================================================================
    # 3. ACTION ROUTER
    # =========================================================================
    
    def check_reminders(self):
        """Tk ticker: announce calendar events coming up, once each.

        Deliberately bypasses the LLM -- no model call, no latency, and no chance of a
        1.5B model garbling the time.
        """
        try:
            # Never talk over a conversation; just wait for the next tick.
            if self.current_state != BotStates.IDLE or self.pending_interjection:
                return

            lead = int(CURRENT_CONFIG.get("reminder_lead_minutes", 30))
            now = datetime.datetime.now(tz=_local_tz())

            for ev in events_for_day("today"):
                if ev["uid"] in self.announced:
                    continue
                mins = (ev["start"] - now).total_seconds() / 60.0
                if 0 < mins <= lead:
                    self.announced.add(ev["uid"])
                    who = CURRENT_CONFIG.get("user_name", "").strip()
                    lead_in = f"{who}! " if who else ""
                    self.pending_interjection = (
                        f"{lead_in}{ev['summary']} in {int(round(mins))} minutes.")
                    print(f"[CAL] reminder due: {self.pending_interjection}", flush=True)
                    self.interject_event.set()
                    break
        except Exception as e:
            print(f"[CAL] reminder check failed: {e}", flush=True)
        finally:
            self.master.after(30_000, self.check_reminders)

    def run_tool(self, name, args):
        """Execute one validated tool call and return a plain-text result.

        Ollama guarantees `name` is one of TOOLS and `args` matches the schema, so
        there is no alias table, no key guessing and no CHAT_FALLBACK rescue here.
        """
        if name == "search_web":
            query = (args or {}).get("query", "").strip()
            if not query:
                return "No search query was given."
            print(f"[TOOL] search_web({query!r})", flush=True)
            try:
                with DDGS() as ddgs:
                    results = []
                    try:
                        results = list(ddgs.text(query, region="us-en", max_results=3))
                    except Exception as e:
                        print(f"[TOOL] text search failed: {e}", flush=True)
                    if not results:
                        try:
                            results = list(ddgs.news(query, region="us-en", max_results=3))
                        except Exception as e:
                            print(f"[TOOL] news search failed: {e}", flush=True)

                    if not results:
                        return f"No results found for '{query}'."

                    lines = []
                    for r in results:
                        title = r.get("title", "")
                        body = r.get("body", r.get("snippet", ""))
                        lines.append(f"- {title}: {body[:220]}")
                    return f"Search results for '{query}':\n" + "\n".join(lines)
            except Exception as e:
                print(f"[TOOL] search error: {e}", flush=True)
                return "The web search failed; the network may be unavailable."

        if name == "get_agenda":
            day = (args or {}).get("day", "today")
            print(f"[TOOL] get_agenda({day!r})", flush=True)
            evs = events_for_day(day)
            if not evs:
                return f"Nothing on the calendar {day}."
            parts = [f"{e['summary']} at {describe_time(e['start'])}" for e in evs]
            return f"Calendar for {day}: " + "; ".join(parts)

        return f"Tool '{name}' is not available."

    def speak_text(self, text):
        """Split a finished reply into sentences and queue them for Piper."""
        for part in re.split(r'(?<=[.!?])\s+', text):
            part = part.strip()
            if part and re.search(r'[a-zA-Z0-9]', part):
                with self.tts_queue_lock:
                    self.tts_queue.append(part)

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================
    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_active.set()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

            follow_up_secs = CURRENT_CONFIG.get("follow_up_seconds", 6)

            while True:
                # --- outer loop: one wake-word session ---
                trigger_source = self.detect_wake_word_or_ptt()

                # A reminder fired. Speak it, then fall through to reopen the wake-word
                # stream -- no conversation follows unless the user says the wake word.
                if trigger_source == "INTERJECT":
                    self.interject_event.clear()
                    line = self.pending_interjection or ""
                    self.pending_interjection = None
                    if line:
                        self.set_state(BotStates.SPEAKING, "Reminder!")
                        self.append_to_text(f"BOT: {line}")
                        self.speak_text(line)
                        self.wait_for_tts()
                    self.set_state(BotStates.IDLE, "Waiting...")
                    continue

                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                follow_up = False
                while True:
                    # --- inner loop: one conversation, no wake word between turns ---
                    if follow_up:
                        # BMO just spoke. Listen briefly for a reply; if the room stays
                        # quiet, fall back to requiring the wake word again.
                        self.set_state(BotStates.LISTENING, "Still listening...")
                        audio_file = self.record_voice_adaptive(wait_for_speech=follow_up_secs)
                    else:
                        self.set_state(BotStates.LISTENING, "I'm listening!")
                        audio_file = (self.record_voice_ptt() if trigger_source == "PTT"
                                      else self.record_voice_adaptive())

                    if not audio_file:
                        if not follow_up:
                            self.set_state(BotStates.IDLE, "Heard nothing.")
                        break

                    user_text = self.transcribe_audio(audio_file)
                    if not user_text:
                        if not follow_up:
                            self.set_state(BotStates.IDLE, "Transcription empty.")
                        break

                    self.append_to_text(f"YOU: {user_text}")
                    self.interrupted.clear()
                    self.chat_and_respond(user_text, img_path=None)

                    # Spacebar interrupt must escape the conversation, not just the reply.
                    if self.interrupted.is_set():
                        self.interrupted.clear()
                        break

                    follow_up = getattr(self, "last_reply_was_question", False)
                    if not follow_up:
                        break                      # straight back to the wake word

                self.set_state(BotStates.IDLE, "Waiting...")

        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up brains...")
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"Failed to load {TEXT_MODEL}: {e}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("Models loaded.", flush=True)

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()

        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        self.oww_model.reset()
        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        # Let ALSA release the PCM from the previous turn (TTS playback / recording) before
        # opening capture. Same guard record_voice_* uses at agent.py:725 / :749.
        # NOTE: this is a STARTUP guard only. We never close and reopen the device to
        # recover from an overflow -- doing that is what produced -9999.
        try:
            sd.stop()
        except Exception:
            pass
        time.sleep(0.2)

        try:
            return self._listen_loop(input_rate)
        except Exception as e:
            print(f"[CRITICAL] Wake Word Stream Error: {e}", flush=True)
            traceback.print_exc()
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

    def _listen_loop(self, input_rate):
        """Callback-driven wake word capture with a ring buffer.

        PortAudio fills audio_q from its own high-priority audio thread; this loop drains
        it. Capture is decoupled from ONNX inference, so slow prediction grows the queue
        instead of overflowing the ALSA ring buffer. An overflow is logged and NEVER tears
        the stream down.
        """
        OWW_RATE = 16000
        OWW_FRAME = 1280        # openWakeWord requires EXACTLY this many samples @ 16 kHz

        # Raw input samples that map to exactly one model frame: 44100 -> 3528, 48000 -> 3840.
        # Deliberately independent of whatever blocksize PortAudio hands us, so an odd or
        # unexpected device block size can no longer distort the model's time base.
        in_per_frame = OWW_FRAME * input_rate // OWW_RATE
        g = math.gcd(OWW_RATE, input_rate)
        up, down = OWW_RATE // g, input_rate // g      # 44100 -> 160/441

        audio_q = queue.Queue(maxsize=64)              # ~1s of slack, bounded
        overflowed = threading.Event()

        def callback(indata, frames, time_info, status):
            if status:
                overflowed.set()                       # consumer logs it; never fatal
            try:
                audio_q.put_nowait(indata.copy())      # PortAudio reuses indata
            except queue.Full:
                pass                                   # drop, rather than stall the audio thread

        stream_args = {
            "samplerate": input_rate,
            "channels": 1,
            "dtype": "int16",
            "blocksize": 0,                            # let PortAudio pick a period ALSA likes
            "device": INPUT_DEVICE_NAME,
            "callback": callback,
        }

        pending = np.zeros(0, dtype=np.int16)

        with sd.InputStream(**stream_args) as stream:
            # Diagnostic: this line definitively explains any future block-size anomaly.
            print(f"[AUDIO] rate={input_rate} stream_block={stream.blocksize} "
                  f"latency={stream.latency} in_per_frame={in_per_frame} "
                  f"resample={up}/{down} -> {OWW_FRAME}", flush=True)

            while True:
                if self.ptt_event.is_set():
                    self.ptt_event.clear()
                    return "PTT"

                # A reminder is due. Exit the same way PTT does so the `with` block
                # closes the stream; detect_wake_word_or_ptt reopens it afterwards
                # with the sd.stop() + settle guard already in place.
                if self.interject_event.is_set():
                    return "INTERJECT"

                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    sys.stdin.readline()
                    return "CLI"

                try:
                    block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if overflowed.is_set():
                    overflowed.clear()
                    print("!", end="", flush=True)

                pending = np.concatenate((pending, block.reshape(-1)))

                while len(pending) >= in_per_frame:
                    raw, pending = pending[:in_per_frame], pending[in_per_frame:]
                    frame = np.clip(scipy.signal.resample_poly(raw, up, down),
                                    -32768, 32767).astype(np.int16)

                    # openWakeWord is a streaming model: feed every consecutive frame,
                    # silence included, or its melspectrogram/embedding history is corrupted.
                    self.oww_model.predict(frame)
                    for mdl in self.oww_model.prediction_buffer.keys():
                        score = list(self.oww_model.prediction_buffer[mdl])[-1]
                        if score > 0.1:
                            print(f"\r[Oww] {mdl}: {score:.3f}   ", end="", flush=True)
                        if score > WAKE_WORD_THRESHOLD:
                            print(f"\n[WAKE] Triggered on '{mdl}' with score: {score:.2f}", flush=True)
                            self.oww_model.reset()
                            return "WAKE"


    def record_voice_adaptive(self, filename="input.wav", wait_for_speech=None):
        """Record until ~1.5s of silence.

        wait_for_speech: if set, give up and return None when speech has not STARTED
        within that many seconds. Used for the follow-up window after BMO speaks, so a
        silent room returns to wake-word mode instead of banking room tone.
        """
        print("Recording (Adaptive)...", flush=True)
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        # Thresholds are MEASURED from the room at the start of every recording, not
        # hardcoded. A fixed 0.006 was tuned once against one ALSA gain in one room and
        # silently drifted out of alignment whenever any of that changed.
        CALIB_CHUNKS = 8                 # ~0.4s; audio is still captured, nothing is lost
        NOISE_FLOOR_MIN, NOISE_FLOOR_MAX = 0.002, 0.025
        calib = []
        # Sensible defaults until calibration completes on the first few chunks.
        silence_threshold = 0.006
        speech_threshold = 0.015
        silence_duration = 1.5
        max_record_time = 30.0
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)

        num_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)
        recorded_chunks = 0
        silence_started = False

        # --- DEBUG INSTRUMENTATION -------------------------------------------------
        # The loop below can only exit on state the CALLBACK mutates (silence_started /
        # recorded_chunks), and max_chunks counts callback INVOCATIONS, not seconds. So
        # if the callback never fires, nothing can ever end the loop. HARD_TIMEOUT is
        # the missing wall-clock guard. Thresholds above are deliberately unchanged.
        HARD_TIMEOUT = 10.0
        last_volume = [0.0]
        speech_started = [False]      # observational only -- gates nothing
        last_status = [None]
        events = []                   # appended by callback, drained by the main loop
        # ---------------------------------------------------------------------------

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            nonlocal silence_threshold, speech_threshold
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1

            # Cheap stores only. Never print from the audio thread: it runs at high
            # priority and I/O here causes glitches/overflows on the Pi.
            last_volume[0] = volume_norm
            if status:
                last_status[0] = str(status)

            # --- calibration window ---
            if recorded_chunks <= CALIB_CHUNKS:
                calib.append(volume_norm)
                if recorded_chunks == CALIB_CHUNKS:
                    # 25th percentile, not the mean: an early word shouldn't inflate the
                    # floor. The clamp stops a calibration taken mid-speech from producing
                    # absurd thresholds.
                    floor = float(np.percentile(calib, 25))
                    floor = max(NOISE_FLOOR_MIN, min(NOISE_FLOOR_MAX, floor))
                    silence_threshold = floor * 1.6
                    speech_threshold = floor * 3.0
                    events.append(f"noise_floor={floor:.4f} "
                                  f"silence<{silence_threshold:.4f} "
                                  f"speech>={speech_threshold:.4f}")
                return

            if not speech_started[0] and volume_norm >= speech_threshold:
                speech_started[0] = True
                events.append("Speech detected")

            if volume_norm < silence_threshold:
                if silent_chunks == 0:
                    events.append("Silence started")
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks:
                    if not silence_started:
                        events.append("Silence duration reached, stopping recording")
                    silence_started = True
            else: silent_chunks = 0

        abandoned = False
        start = time.time()
        try:
            # Explicitly close stream if it exists to free hardware
            sd.stop()
            time.sleep(0.2)

            start = time.time()
            last_log = -1.0
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,
                                device=INPUT_DEVICE_NAME, blocksize=chunk_size) as stream:
                print(f"[REC] stream open: rate={samplerate} req_block={chunk_size} "
                      f"actual_block={stream.blocksize} latency={stream.latency} "
                      f"dtype=float32(default) threshold={silence_threshold}", flush=True)

                while not silence_started and recorded_chunks < max_chunks:
                    sd.sleep(int(chunk_duration * 1000))
                    elapsed = time.time() - start

                    while events:
                        print(f"[REC] {events.pop(0)}", flush=True)

                    if elapsed - last_log >= 0.5:
                        last_log = elapsed
                        print(f"[REC] volume={last_volume[0]:.5f} "
                              f"speech_started={speech_started[0]} "
                              f"silence_timer={silent_chunks * chunk_duration:.2f}s "
                              f"buffer_frames={recorded_chunks} "
                              f"elapsed={elapsed:.1f}s"
                              + (f" status={last_status[0]}" if last_status[0] else ""),
                              flush=True)

                    # Hard ceiling on a follow-up attempt regardless of speech_started,
                    # so a noise blip costs a few seconds at most, never the full timeout.
                    if wait_for_speech and elapsed >= wait_for_speech + 5:
                        print(f"[REC] follow-up window overran -> back to wake word", flush=True)
                        abandoned = True
                        break

                    if wait_for_speech and not speech_started[0] and elapsed >= wait_for_speech:
                        print(f"[REC] no follow-up within {wait_for_speech}s "
                              f"-> back to wake word", flush=True)
                        abandoned = True
                        break

                    if elapsed >= HARD_TIMEOUT:
                        print(f"[REC] HARD TIMEOUT {HARD_TIMEOUT}s reached "
                              f"(callbacks={recorded_chunks}) - stopping", flush=True)
                        break

                if recorded_chunks >= max_chunks:
                    print("[REC] Max recording duration reached", flush=True)
        except Exception as e:
            print(f"[AUDIO ERROR] Adaptive Recording Failed: {e}", flush=True)
            return None

        while events:
            print(f"[REC] {events.pop(0)}", flush=True)
        if abandoned:
            return None

        print(f"[REC] exiting: chunks={recorded_chunks} silence_started={silence_started} "
              f"elapsed={time.time() - start:.1f}s -> save_audio_buffer", flush=True)
        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.5)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        buffer = []
        def callback(indata, frames, time_info, status): buffer.append(indata.copy())
        
        try:
            # Explicitly close stream if it exists to free hardware
            # This is critical on Pi 5 where hardware contention causes freezes
            sd.stop() 
            time.sleep(0.2)
            
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set(): 
                    sd.sleep(50)
        except Exception as e: 
            print(f"[AUDIO ERROR] PTT Recording Failed: {e}", flush=True)
            return None
            
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0

        # whisper.cpp REQUIRES 16 kHz mono WAV and rejects any other rate outright --
        # the complaint goes to stderr and stdout comes back empty, which surfaced as
        # Heard: ''. The mic captures at 44100, so resample before writing.
        WHISPER_RATE = 16000
        if samplerate != WHISPER_RATE:
            g = math.gcd(WHISPER_RATE, samplerate)
            audio_data = scipy.signal.resample_poly(audio_data, WHISPER_RATE // g, samplerate // g)
            samplerate = WHISPER_RATE

        audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        print(f"[REC] saved={os.path.abspath(filename)} "
              f"duration={len(audio_data) / float(samplerate):.2f}s "
              f"peak={peak:.4f} rate={samplerate}", flush=True)
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        try:
            result = subprocess.run(
                ["./whisper.cpp/build/bin/whisper-cli", "-m", "./whisper.cpp/models/ggml-base.en.bin", "-l", "en", "-t", "4", "-f", filename],
                capture_output=True, text=True
            )
            transcription_lines = result.stdout.strip().split('\n')
            if transcription_lines and transcription_lines[-1].strip():
                last_line = transcription_lines[-1].strip()
                if ']' in last_line: transcription = last_line.split("]")[1].strip()
                else: transcription = last_line
            else: transcription = ""

            # stderr was always captured here but discarded, so ANY whisper failure
            # (missing binary, missing model, bad rate) looked identical to silence.
            if result.returncode != 0 or not transcription:
                print(f"[STT] rc={result.returncode}", flush=True)
                err = (result.stderr or "").strip()
                if err:
                    print(f"[STT] stderr: {err[-500:]}", flush=True)

            print(f"Heard: '{transcription}'", flush=True)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Watching...")
        try:
            subprocess.run(["rpicam-still", "-t", "500", "-n", "--width", "640", "--height", "480", "-o", BMO_IMAGE_FILE], check=True)
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = Image.open(BMO_IMAGE_FILE)
                img = img.rotate(rotation, expand=True) 
                img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def chat_and_respond(self, text, img_path=None, _depth=0):
        if "forget everything" in text.lower() or "reset memory" in text.lower():
            self.session_memory = []
            self.permanent_memory = []
            self.save_chat_history()
            with self.tts_queue_lock:
                self.tts_queue.append("Okay. Memory wiped.")
            self.set_state(BotStates.IDLE, "Memory Wiped")
            return

        model_to_use = VISION_MODEL if img_path else TEXT_MODEL
        self.set_state(BotStates.THINKING, "Thinking...", cam_path=img_path)

        # System message is rebuilt here every turn and never read from memory.json.
        system_msg = {"role": "system", "content": build_system_prompt()}
        if img_path:
            # moondream is a small VLM that expects roughly image + prompt. Handing it
            # the BMO persona (tool rules, date, name) is out of distribution and it
            # returns EMPTY content -- which surfaced as "BMO's brain went quiet".
            # So: no system message here. moondream describes, BMO re-voices below.
            messages = [{"role": "user",
                         "content": f"{text}\n\nAnswer in one short sentence.",
                         "images": [img_path]}]
        else:
            # Cap the working context. permanent_memory (up to 12 turns from disk) plus
            # an UNBOUNDED session_memory was reaching 18+ messages for a 4-word question;
            # on a Pi CPU prompt processing then dominates the whole response time.
            history = (self.permanent_memory + self.session_memory)[-MEMORY_TURNS:]
            messages = [system_msg] + history + [{"role": "user", "content": text}]

        # --- calendar: answered entirely without the model ---
        if not img_path and CALENDAR_RE.search(text):
            day = "tomorrow" if "tomorrow" in text.lower() else "today"
            print(f"[PREROUTE] calendar -> get_agenda({day})", flush=True)
            events = events_for_day(day)
            print(f"[PREROUTE] {len(events)} events, answering without the model", flush=True)

            final_text = format_agenda(events, day,
                                       CURRENT_CONFIG.get("user_name", "").strip())
            self.set_state(BotStates.SPEAKING, "Speaking...")
            self.append_to_text("BOT: ", newline=False)
            self.append_to_text(final_text, newline=True)
            self.speak_text(final_text)
            self.session_memory.append({"role": "user", "content": text})
            self.session_memory.append({"role": "assistant", "content": final_text})
            self.last_reply_was_question = False
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")
            return

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        try:
            print(f"[LLM] model={model_to_use} turns={len(messages)} prompt={text!r}", flush=True)

            # Pass 1: the model either answers, or asks for a tool. The vision model
            # has no tool template, so tools are only offered on the text path.
            kwargs = {"model": model_to_use, "messages": messages, "options": OLLAMA_OPTIONS}
            if not img_path:
                kwargs["tools"] = TOOLS
            _t0 = time.time()
            resp = ollama.chat(**kwargs)
            msg = resp["message"]
            print(f"[LLM] pass1 {time.time() - _t0:.1f}s", flush=True)

            calls = msg.get("tool_calls") or []
            print(f"[ROUTER] tool_calls={[c['function']['name'] for c in calls]}", flush=True)

            if calls:
                # The camera tool re-runs the whole turn against the vision model.
                for call in calls:
                    if call["function"]["name"] == "look" and _depth == 0:
                        self.thinking_sound_active.clear()
                        new_img = self.capture_image()
                        if new_img:
                            self.chat_and_respond(text, img_path=new_img, _depth=1)
                            return

                messages.append(msg)
                for call in calls:
                    name = call["function"]["name"]
                    args = call["function"]["arguments"] or {}
                    print(f"[ROUTER] executing {name}({args})", flush=True)
                    result = self.run_tool(name, args)
                    messages.append({"role": "tool", "name": name, "content": str(result)})

                # Pass 2: let BMO phrase the tool result in his own voice.
                self.set_state(BotStates.THINKING, "Reading...", cam_path=img_path)
                _t1 = time.time()
                resp = ollama.chat(model=model_to_use, messages=messages,
                                   options=OLLAMA_OPTIONS)
                msg = resp["message"]
                print(f"[LLM] pass2 {time.time() - _t1:.1f}s", flush=True)

            final_text = (msg.get("content") or "").strip()

            # moondream captions like a caption model. Re-voice it as BMO so the
            # camera path sounds like the rest of him.
            if img_path and final_text:
                try:
                    revoice = ollama.chat(
                        model=TEXT_MODEL,
                        messages=[{"role": "system", "content": build_system_prompt()},
                                  {"role": "user",
                                   "content": f"You just looked through your camera and saw: "
                                              f"{final_text}\n\nThe question was: {text}\n"
                                              f"Say what you see, in your own voice, one short sentence."}],
                        options=OLLAMA_OPTIONS)
                    voiced = (revoice["message"].get("content") or "").strip()
                    if voiced:
                        print(f"[VISION] raw={final_text!r}", flush=True)
                        final_text = voiced
                except Exception as e:
                    print(f"[VISION] re-voice failed, using raw: {e}", flush=True)

            if not final_text:
                final_text = "BMO's brain went quiet. Ask me again?"


            # Only keep the mic open if BMO actually needs an answer back. Lingering
            # after every reply made him feel like he never went to sleep.
            self.last_reply_was_question = final_text.rstrip().endswith("?")

            print(f"[LLM] response={final_text!r} "
                  f"question={self.last_reply_was_question}", flush=True)

            self.thinking_sound_active.clear()
            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=img_path)
            self.append_to_text("BOT: ", newline=False)
            self.append_to_text(final_text, newline=True)
            self.speak_text(final_text)

            self.session_memory.append({"role": "user", "content": text})
            self.session_memory.append({"role": "assistant", "content": final_text})

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Ready")

        except Exception as e:
            self.thinking_sound_active.clear()
            print(f"LLM Error: {e}", flush=True)
            traceback.print_exc()
            self.set_state(BotStates.ERROR, "Brain Freeze!")

    def wait_for_tts(self):
        while self.tts_queue or self.tts_active.is_set():
            if self.interrupted.is_set(): break
            time.sleep(0.1)

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue: 
                    text = self.tts_queue.pop(0)
                    self.tts_active.set() 
            if text: 
                self.speak(text)
                self.tts_active.clear() 
            else: time.sleep(0.05)

    def speak(self, text):
        clean = re.sub(r"[^\w\s,.!?:-]", "", text)
        if not clean.strip(): return
        
        print(f"[PIPER SPEAKING] '{clean}'", flush=True)
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")
        
        try:
            self.current_audio_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw"], 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            
            self.current_audio_process.stdin.write(clean.encode() + b'\n')
            self.current_audio_process.stdin.close() 

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000 

            PIPER_RATE = 22050
            use_native_rate = False
            
            try:
                sd.check_output_settings(device=None, samplerate=PIPER_RATE)
            except:
                use_native_rate = True

            with sd.RawOutputStream(samplerate=native_rate if use_native_rate else PIPER_RATE, 
                                    channels=1, dtype='int16', 
                                    device=None, latency='low', blocksize=2048) as stream:
                while True:
                    if self.interrupted.is_set(): break
                    data = self.current_audio_process.stdout.read(4096)
                    if not data: break 
                    
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if len(audio_chunk) > 0:
                        self.current_volume = np.max(np.abs(audio_chunk))
                        if use_native_rate:
                            num_samples = int(len(audio_chunk) * (native_rate / PIPER_RATE))
                            audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.int16)
                        stream.write(audio_chunk.tobytes())
                    else:
                        self.current_volume = 0
                time.sleep(0.5) 
                    
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0 
            if self.current_audio_process:
                if self.current_audio_process.stdout: self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None: self.current_audio_process.terminate()
                self.current_audio_process = None

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound: self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set(): return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path): return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000 

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait() 
        except: pass

    def load_chat_history(self):
        """Load prior user/assistant turns, discarding anything from an older schema.

        Stale turns are actively harmful: the model reads its own past answers as
        context and repeats them. Gemma-era history is why BMO still said "Sparky"
        and "November 2023" after the persona was replaced -- a correct injected date
        cannot outvote twelve turns of history asserting otherwise.
        """
        if not os.path.exists(MEMORY_FILE):
            return []
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[MEM] unreadable ({e}); starting fresh", flush=True)
            return []

        # Legacy files are a bare list -> version 0 -> discarded.
        version = data.get("version", 0) if isinstance(data, dict) else 0
        turns = data.get("turns", []) if isinstance(data, dict) else data

        if version != MEMORY_VERSION:
            print(f"[MEM] discarded incompatible history "
                  f"(v{version}, need v{MEMORY_VERSION}, {len(turns)} turns)", flush=True)
            return []

        turns = [m for m in turns if m.get("role") in ("user", "assistant")]
        print(f"[MEM] loaded {len(turns)} turns (v{version})", flush=True)
        return turns

    def save_chat_history(self):
        conv = [m for m in (self.permanent_memory + self.session_memory)
                if m.get("role") in ("user", "assistant")]
        with open(MEMORY_FILE, "w") as f:
            json.dump({"version": MEMORY_VERSION, "turns": conv[-12:]}, f, indent=2)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    # If this stamp is wrong, the Pi clock is wrong (no RTC battery) -- no code
    # change fixes that. Check `date` and `sudo timedatectl set-ntp true`.
    print(f"[LLM] model={TEXT_MODEL} vision={VISION_MODEL}", flush=True)
    print(f"[LLM] system date stamp: {build_system_prompt().splitlines()[-1]}", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
