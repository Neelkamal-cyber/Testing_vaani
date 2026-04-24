import threading
import time
import os
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import av
from gtts import gTTS
from twilio.rest import Client
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model
from streamlit_autorefresh import st_autorefresh

# --- OS Level Optimizations ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

# --- Constants & Page Config ---
st.set_page_config(page_title="VAANI · Sign Language AI", layout="wide")
MODEL_PATH = "vaani_endec_deploy.h5"
SEQUENCE_LENGTH = 60

# --- WebRTC TURN Server Configuration ---
@st.cache_data(show_spinner=False)
def get_ice_servers():
    """Retrieve Twilio TURN credentials from st.secrets, falling back to STUN."""
    try:
        tw_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        tw_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(tw_sid, tw_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception:
        # Warning: STUN fallback will fail on strict cloud deployments
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIG = RTCConfiguration({"iceServers": get_ice_servers()})

# --- Model Loading ---
@st.cache_resource(show_spinner="⏳ Loading VAANI model…")
def load_resources():
    # Model loading logic remains identical to user's optimal implementation
    model = load_model(MODEL_PATH, compile=False)
    model.predict(np.zeros((1, SEQUENCE_LENGTH, 258)), verbose=0) # Warm-up
    idx_to_word = {0: "Hello", 1: "Thanks", 2: "I Love You"} # Simplified for brevity
    return model, idx_to_word

model, idx_to_word = load_resources()

# --- Thread-Safe Processor Class ---
class VANIProcessor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.sequence = []
        self.predictions = []
        
        # Thread-safe state variables guarded by a mutex lock
        self.lock = threading.Lock()
        self.current_word = ""
        self.sentence = []
        self.frames_since_sign = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True

        # Assume extract_features() processes the results correctly
        # kp = extract_features(results)
        kp = np.zeros(258) # Placeholder for narrative
        self.sequence.append(kp)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:]

        if len(self.sequence) == SEQUENCE_LENGTH:
            inp = np.expand_dims(self.sequence, axis=0)
            res = model.predict(inp, verbose=0)
            pi = int(np.argmax(res))
            conf = float(res[pi])

            if conf > 0.90:
                self.predictions.append(pi)
                self.predictions = self.predictions[-15:]

                if self.predictions.count(pi) > 10:
                    word = idx_to_word.get(pi, "?")
                    # Safely mutate state using the lock
                    with self.lock:
                        if word != self.current_word:
                            self.current_word = word
                            self.frames_since_sign = 0
                            if not self.sentence or self.sentence[-1] != word:
                                self.sentence.append(word)
                                self.sentence = self.sentence[-8:]
            else:
                with self.lock:
                    self.frames_since_sign += 1

        # Safely read state for rendering onto the CV2 frame
        with self.lock:
            if self.frames_since_sign > 60:
                self.current_word = ""
            display_word = self.current_word
            display_sentence = " ".join(self.sentence)

        cv2.putText(img, f"VAANI | {display_word}", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 149), 2)
        cv2.putText(img, display_sentence, (14, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (200, 255, 235), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI and Synchronization ---
st.title("VAANI · Real-Time Sign Language Recognition")

ctx = webrtc_streamer(
    key="vaani",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VANIProcessor,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    async_processing=True,
)

# Auto-refresh the UI every 1000ms to read from the processor thread
# This definitively eliminates the destructive `while True: st.rerun()` loop
st_autorefresh(interval=1000, key="data_refresh")

if ctx.state.playing and ctx.video_processor:
    # Safely extract variables from the processor instance
    with ctx.video_processor.lock:
        synced_sentence = ctx.video_processor.sentence
        synced_word = ctx.video_processor.current_word
        
    st.markdown(f"**Current Sign:** {synced_word}")
    st.markdown(f"**Sentence:** {' '.join(synced_sentence)}")
