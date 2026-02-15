import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import cv2
import tempfile
import os

# --- Page Config ---
st.set_page_config(page_title="Local Video AI", page_icon="🕵️")

st.title("🕵️ Local Video AI (No Keys Needed)")
st.caption("Running 'Moondream' locally on your Codespace CPU")

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    with st.spinner("Downloading AI Model (this happens only once)..."):
        # We use Moondream2, a tiny but powerful vision model
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            revision=revision,
            torch_dtype=torch.float32  # Use float32 for CPU compatibility
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return model, tokenizer

try:
    model, tokenizer = load_model()
    st.success("AI Brain Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Helper: Extract Frame ---
def get_frame(video_path, timestamp_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Convert Color (OpenCV is BGR, Pillow needs RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None

# --- UI Layout ---
video_file = st.file_uploader("Upload MP4", type=["mp4", "mov"])

if video_file:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Create a column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.video(video_path)
        
        # Get video duration (rough estimate)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 10
        cap.release()

        # Timestamp Slider
        timestamp = st.slider("Select time to analyze (seconds)", 0.0, float(duration), 0.0)
        
        if st.button("👀 Analyze This Moment"):
            frame = get_frame(video_path, timestamp)
            if frame:
                st.session_state.current_image = frame
                # Reset chat when looking at a new frame
                st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": f"I am looking at the video frame at {timestamp} seconds. Ask me anything!"})
            else:
                st.error("Could not grab frame.")

    with col2:
        if st.session_state.current_image:
            st.image(st.session_state.current_image, caption="AI's Current View", use_container_width=True)
            
            # Chat Interface
            chat_container = st.container(height=400)
            for msg in st.session_state.messages:
                with chat_container.chat_message(msg["role"]):
                    st.write(msg["content"])

            if prompt := st.chat_input("Ask about this frame..."):
                # User Message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container.chat_message("user"):
                    st.write(prompt)

                # AI Response
                with chat_container.chat_message("assistant"):
                    with st.spinner("Analyzing pixels..."):
                        # Prepare input for Moondream
                        enc_image = model.encode_image(st.session_state.current_image)
                        answer = model.answer_question(enc_image, prompt, tokenizer)
                        
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("👈 Use the slider and click 'Analyze This Moment' to start chatting.")
