import streamlit as st
import requests
import json
import uuid
import base64
import os

# Set page config
st.set_page_config(layout='wide', page_title="OptiChat")

st.title("OptiChat: Talk to your Optimization Model")

API_BASE_URL = "http://localhost:8000"
APP_NAME = "optichat"

# --- Sidebar ---
METADATA_PATH = "tmp/model_objects/metadata.json"

def load_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.sidebar.error(f"Error loading metadata: {e}")
    return {}

metadata = load_metadata()

selected_models = []

if metadata:
    st.sidebar.subheader("Select Model")
    
    selected_models = []
    
    all_dates = sorted(list(metadata.keys()), reverse=True)
    
    for date in all_dates:
        with st.sidebar.expander(f"📅 {date}", expanded=False):
            base_models_data = metadata[date]
            
            for base_model_name in sorted(base_models_data.keys()):
                st.markdown(f"**📓 {base_model_name}**")
                

                base_key = f"chk_{date}_{base_model_name}_base"
                if st.checkbox(f"{base_model_name} (Base)", key=base_key):
                    selected_models.append(base_model_name)
                
                model_info = base_models_data[base_model_name]
                if "modified_models" in model_info:
                    for mod_name in sorted(model_info["modified_models"].keys()):
                        mod_key = f"chk_{date}_{base_model_name}_{mod_name}"
                        if st.checkbox(mod_name, key=mod_key):
                            selected_models.append(mod_name)
                
                st.divider()

    st.session_state.selected_models = selected_models
else:
    st.sidebar.info("No models found. Upload a config to get started.")


st.sidebar.subheader("Load Model Config")
uploaded_json = st.sidebar.file_uploader("Upload JSON Config", type=["json"])


show_model_representation = st.sidebar.checkbox("Show Model Representation", False)
model_representation_placeholder = st.empty()

show_code = st.sidebar.checkbox("Show Code", False)
code_placeholder = st.empty()

show_tech_feedback = st.sidebar.checkbox("Show Technical Feedback", False)
tech_feedback_placeholder = st.empty()

if "messages" in st.session_state:
    chat_history_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
    st.sidebar.download_button(label="Export Chat History", data=chat_history_text, file_name='chat_history.txt', mime='text/plain')

# --- Session Setup ---
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    try:
        url = f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions"
        payload = {
            "session_id": st.session_state.session_id
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"Session created: {st.session_state.session_id}")
        else:
            st.error(f"Failed to create session: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to API server: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper to fetch session state ---
def fetch_session_state():
    try:
        url = f"{API_BASE_URL}/apps/{APP_NAME}/users/{st.session_state.user_id}/sessions/{st.session_state.session_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

# --- JSON Upload Handling ---
if uploaded_json is not None:
    if "last_uploaded_json" not in st.session_state or st.session_state.last_uploaded_json != uploaded_json.name:
        st.session_state.last_uploaded_json = uploaded_json.name
        
        json_content = uploaded_json.read()
        json_base64 = base64.b64encode(json_content).decode('utf-8')
        
        try:
            url = f"{API_BASE_URL}/run"
            payload = {
                "app_name": APP_NAME,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "new_message": {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Initialize session with config",
                            "inline_data": {
                                "mime_type": "application/json",
                                "data": json_base64
                            }
                        }
                    ]
                }
            }
            
            with st.spinner("Initializing session with uploaded config..."):
                response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                st.sidebar.success(f"Successfully uploaded {uploaded_json.name}")
                events = response.json()
                for event in events:
                    if 'content' in event and event['content']:
                        content = event['content']
                        if 'parts' in content:
                            for part in content['parts']:
                                if 'text' in part:
                                    st.session_state.messages.append({"role": "assistant", "content": part['text']})
            else:
                st.sidebar.error(f"Failed to upload config: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.sidebar.error(f"Error uploading config: {e}")

# --- Visualization Rendering ---
session_data = fetch_session_state()
if session_data and "state" in session_data:
    state = session_data["state"]
    
    if show_model_representation:
        if "MODELS_DICTIONARY" in state:
            with model_representation_placeholder.container():
                st.json(state["MODELS_DICTIONARY"])
    
    if show_code:
        if "CFG" in state and "models_code" in state["CFG"]:
             try:
                 paths = state["CFG"]["models_code"].get("local_resources", [])
                 code_content = ""
                 for path in paths:
                     import os
                     if os.path.exists(path):
                         with open(path, "r") as f:
                             code_content += f"# File: {path}\n{f.read()}\n\n"
                 if code_content:
                     with code_placeholder.container():
                         st.code(code_content)
             except Exception:
                 pass

    # Technical Feedback
    if show_tech_feedback:
        pass
else:
    if show_model_representation:
        model_representation_placeholder.info("No model data available yet.")
    if show_code:
        code_placeholder.info("No code available yet.")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare context with selected models
    if "selected_models" in st.session_state and st.session_state.selected_models:
        model_list_str = ", ".join(st.session_state.selected_models)
        context_message = f"[Using model: {model_list_str}] {prompt}"
    else:
        context_message = prompt

    try:
        url = f"{API_BASE_URL}/run"
        payload = {
            "app_name": APP_NAME,
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id,
            "new_message": {
                "role": "user",
                "parts": [
                    {
                        "text": context_message
                    }
                ]
            }
        }
        
        with st.spinner("Thinking..."):
            response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            events = response.json()
            full_response = ""
            for event in events:
                if 'content' in event and event['content']:
                    content = event['content']
                    if 'parts' in content:
                        for part in content['parts']:
                            if 'text' in part:
                                full_response += part['text']
            
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    st.markdown(full_response)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to API server: {e}")
