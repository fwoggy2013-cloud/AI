import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from gtts import gTTS
import torch
import cv2
import tempfile
import os

# --- Page Config ---
st.set_page_config(page_title="GHOST AI", page_icon="👻", layout="wide")

# --- Custom CSS for "Aura" ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #333;
        color: white;
        border: 1px solid #555;
    }
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Local Brain (Moondream2) ---
# This downloads the model ONCE to the server, bypassing school wifi blocks
@st.cache_resource
def load_model():
    with st.spinner("👻 Summoning the AI Brain... (This takes 1 minute)"):
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        
        # Load the model into the Codespace memory (CPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            revision=revision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return model, tokenizer

# --- Voice Function ---
def speak(text):
    try:
        # Create audio file from text
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            # Display audio player that auto-plays
            st.audio(fp.name, format="audio/mp3", start_time=0)
    except Exception as e:
        st.warning("Voice disabled (Network issue).")

# --- Main App ---
st.title("👻 GHOST AI")
st.caption("Running Locally • No API Keys • Unblockable")

# Load the brain
try:
    model, tokenizer = load_model()
    st.success("System Online.")
except Exception as e:
    st.error(f"Failed to load AI: {e}")
    st.stop()

# --- Tabs ---
tab1, tab2 = st.tabs(["🎥 Video Analyst", "👁️ Live Vision"])

# --- TAB 1: VIDEO ---
with tab1:
    video_file = st.file_uploader("Upload MP4 Video", type=['mp4', 'mov'])
    
    if video_file:
        # Save temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
            
            # Slider to pick a specific moment
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            timestamp = st.slider("Select Time to Analyze (Seconds)", 0.0, duration, 0.0)
            
        with col2:
            if st.button("Analyze This Moment"):
                # specific frame extraction
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                
                if ret:
                    # Convert color for AI
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    st.image(pil_image, caption="AI Vision Feed", use_container_width=True)
                    
                    with st.spinner("Thinking..."):
                        # AI Processing
                        enc_image = model.encode_image(pil_image)
                        # We prompt the AI to describe it
                        answer = model.answer_question(enc_image, "Describe this scene in detail.", tokenizer)
                        
                        st.markdown("### AI Output:")
                        st.write(answer)
                        speak(answer)
                else:
                    st.error("Could not grab frame.")

# --- TAB 2: LIVE CAMERA ---
with tab2:
    st.write("Take a snapshot. The AI will see what you see.")
    
    cam_image = st.camera_input("Camera Feed")
    
    if cam_image:
        img = Image.open(cam_image)
        
        prompt = st.text_input("Ask the AI something:", value="What is happening in this image?")
        
        if st.button("Process Snapshot"):
            with st.spinner("Analyzing..."):
                enc_image = model.encode_image(img)
                answer = model.answer_question(enc_image, prompt, tokenizer)
                
                st.markdown("### AI Output:")
                st.write(answer)
                speak(answer)
