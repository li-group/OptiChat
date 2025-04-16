import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import os
from io import StringIO
import time
import tempfile
import io
from extractor import initial_loading
from extractor import update_model_representation, get_skipJSON, feed_skipJSON
from utils import get_agents
from utils import OptiChat_workflow_exp
from pyomo.opt import TerminationCondition
import json


#===============string generator for LLM's response=================

# display a long string of response from GPT in chunks in a user interface
def string_generator(long_string, chunk_size=50):
    for i in range(0, len(long_string), chunk_size):
        yield long_string[i:i+chunk_size]
        time.sleep(0.1)  

# display a long string of response from Gemini in chunks in a user interface
# Response from Gemini is "text" format, diff from GPT so we need a different function
def gemini_stream_generator(response):
    for chunk in response:
        if chunk.text:
            string_chunk=str(chunk.text)
            yield string_chunk
            time.sleep(0.05)  # Add a small delay for smoother display

#==================================================================

        
st.session_state['temperature'] = 0.1  # add default temp to control the randomness of model's output
st.session_state['json_mode'] = True  # application can handle output data in JSON
st.session_state['illustration_stream'] = True  # streams illustrations
st.session_state['inference_stream'] = True  # application streams the results of an inference process
st.session_state['explanation_stream'] = True  # application streams explanations or interpretations of the results
st.session_state['internal_experiment'] = False  
st.session_state['external_experiment'] = False  



# confgiure layout and title of your application page
st.set_page_config(layout='wide')
st.title("OptiChat: Talk to your Optimization Model")

# drop down option on application for user to select a model from list of models
llm_model = st.sidebar.selectbox(
    label="LLM-Model", 
    options=["gpt-4-turbo-preview", "gpt-4-turbo", "gpt-4-1106-preview", 
             "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", 
             "gemini-1.5-flash"], )

#======setting up access token to use API===============

# setting up API key for GPT
if llm_model.startswith("gpt"):
    st.session_state["llm_model"] = llm_model
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    st.session_state['client'] = client
    
# setting up API key for Gemini
elif llm_model.startswith("gemini"):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client= genai.GenerativeModel(llm_model)
    st.session_state['llm_model'] =  client
    st.session_state['client'] = client
    
else:
    print("select a model")
print(f"Client object: {client}")
print(f"Client type: {type(client)}")

#==================================================
    

# Set a default model when application starts
if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "gpt-4-turbo-preview"
if "models_dict" not in st.session_state:
    st.session_state["models_dict"] = {"model_representation": {}}
if "code" not in st.session_state:
    st.session_state["code"] = ""


st.sidebar.subheader("Load Pyomo File")
uploaded_file = st.sidebar.file_uploader("Upload Model", type=["py"])
uploaded_json = st.sidebar.file_uploader("Upload JSON", type=["json"])
st.session_state['py_path'] = None
st.session_state['fn_names'] = ["feasibility_restoration",
                                "sensitivity_analysis",
                                "components_retrival",
                                "evaluate_modification",
                                "external_tools"]

interpreter, explainer, engineer, coordinator = get_agents(st.session_state.fn_names,
                                                           st.session_state.client,
                                                           st.session_state.llm_model)
st.session_state['Interpreter'] = interpreter
st.session_state['Explainer'] = explainer
st.session_state['Engineer'] = engineer
st.session_state['Coordinator'] = coordinator


if not st.session_state.get("messages"):
    st.session_state["messages"] = []

if not st.session_state.get("team_conversation"):
    st.session_state["team_conversation"] = []

if not st.session_state.get("chat_history"):
    st.session_state["chat_history"] = []

if not st.session_state.get("detailed_chat_history"):
    st.session_state["detailed_chat_history"] = []


def process():
    #make sure user uploads both model and json file
    if uploaded_file is None or uploaded_json is None:
        st.error("Please upload both the Pyomo model and the corresponding JSON file.")
        return
    
    #load json
    json_data = json.load(uploaded_json)
    models_dict, code = initial_loading(uploaded_file, is_uploaded=True, json_data=json_data)

    #models_dict, code = initial_loading(uploaded_file)

    # user_prompt= st.chat_input("I have uploaded a Pyomo model.")
    # if user_prompt:
    #     st.chat_message("user").markdown("user_prompt")
   
    st.session_state.messages.append({"role": "user", "content": "I have uploaded a Pyomo model."})
    # interpret the model components
    models_dict, cnt, completion = st.session_state.Interpreter.generate_interpretation_exp(st.session_state,
                                                                                            models_dict, code)
    st.session_state['models_dict'] = models_dict
    st.session_state['code'] = code
    # update model representation with component descriptions
    update_model_representation(st.session_state.models_dict)
    
#===========model illustration ====================================
    
    # illustration for GPT
    if llm_model.startswith("gpt"):
        illustration_stream = st.session_state.Interpreter.generate_illustration_exp(st.session_state,
                                                                                 models_dict["model_representation"])
        # print(type(illustration_stream))
        with st.chat_message("assistant"):
            illustration = st.write_stream(illustration_stream)
     
    # illustration for Gemini  
    elif llm_model.startswith("gemini"):
        try:
            illustration_stream = st.session_state.Interpreter.generate_illustration_exp(
                st.session_state, models_dict["model_representation"]
            )

            if illustration_stream is None:
                st.error("Gemini API error.")
                illustration = "Gemini Error"
            else:
                full_illustration_response = ""  # Hold the full response content

                with st.chat_message("assistant"):
                    placeholder = st.empty() # Create a placeholder to update live
                    
                    for chunk in illustration_stream:                      
                        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)# Extract text or data from the chunk
                        full_illustration_response += chunk_text                       
                        placeholder.markdown(full_illustration_response) # Update the content in the placeholder, progressively
                        time.sleep(0.01)  # Adjust this for a smoother experience

                illustration = full_illustration_response  # Store the complete content
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            illustration = "Gemini Error"
    
                
#============================================================================       


    # update model representation with model description
    st.session_state.models_dict['model_1']['model description'] = illustration
    update_model_representation(st.session_state.models_dict)
    
#======if model is infeasible then generate inference========================
    
    if st.session_state.models_dict['model_1']['model status'] in [TerminationCondition.infeasible,
                                                                   TerminationCondition.infeasibleOrUnbounded]:
        # inference for GPT
        if llm_model.startswith("gpt"):
            inference_stream = st.session_state.Interpreter.generate_inference_exp(st.session_state,
                                                                                    models_dict["model_representation"])
            # print(type(illustration_stream))
            with st.chat_message("assistant"):
                inference = st.write_stream(inference_stream)
        
        # inference for Gemini       
        elif llm_model.startswith("gemini"):
            try:
                inference_stream = st.session_state.Interpreter.generate_inference_exp(
                    st.session_state, models_dict["model_representation"]
                )

                if inference_stream is None:
                    st.error("Gemini API error.")
                    inference = "Gemini Error"
                else:
                    full_inference_response = ""  # Hold the full response content
                    with st.chat_message("assistant"):    
                        placeholder = st.empty() # Create a placeholder to update live
                        
                        for chunk in inference_stream:
                            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)   # Extract text or data from the chunk
                            full_inference_response += chunk_text
                            placeholder.markdown(full_inference_response)
                            time.sleep(0.01)  # Adjust this for a smoother experience

                    inference = full_illustration_response  # Store the complete content
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                inference = "Gemini Error"
            
       
        # update model representation with inference description
        st.session_state.models_dict['model_1']['model description'] = illustration + '\n' + inference
        update_model_representation(st.session_state.models_dict)
        
#=============================================================================
    

    # append model representation to messages
    st.session_state.messages.append({"role": "assistant",
                                      "content": st.session_state.models_dict["model_representation"]["model description"]})

    # append detailed chat history
    st.session_state.chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.chat_history.append("assistant: " +
                                         st.session_state.models_dict["model_representation"]["model description"])
    st.session_state.detailed_chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.detailed_chat_history.append("assistant: " +
                                                  st.session_state.models_dict["model_representation"]["model description"])

    # save model_description and description of every component
    if not os.path.exists("logs/model_json"):
        os.makedirs("logs/model_json")
    if not os.path.exists("logs/code_draft"):
        os.makedirs("logs/code_draft")
    if not os.path.exists("logs/ilps"):
        os.makedirs("logs/ilps")

    json2save = get_skipJSON(st.session_state.models_dict["model_representation"])
    with open(f"logs/model_json/{os.path.splitext(uploaded_file.name)[0]}.json", "w") as f:
        json.dump(json2save, f)


def load_json():
    if uploaded_file is None or uploaded_json is None:
        st.error("Please upload both the Pyomo model and the corresponding JSON file.")
        return

    # models_dict, code = initial_loading(uploaded_file)
    json_data = json.load(uploaded_json)
    models_dict, code = initial_loading(uploaded_file, is_uploaded=True, json_data=json_data)

    with st.chat_message("user"):
        st.markdown("I have uploaded a Pyomo model.")
    st.session_state.messages.append({"role": "user", "content": "I have uploaded a Pyomo model."})

    skipJSON = json_data
    models_dict = feed_skipJSON(skipJSON, models_dict)

    st.session_state["models_dict"] = models_dict
    st.session_state['code'] = code
    # update model representation with component and model descriptions
    update_model_representation(st.session_state.models_dict)

    time.sleep(8)
    stream = string_generator(skipJSON["model description"])
    with st.chat_message("assistant"):
        st.write_stream(stream)

    # append model representation to messages
    st.session_state.messages.append({"role": "assistant",
                                      "content": st.session_state.models_dict["model_representation"][
                                          "model description"]})

    # append detailed chat history
    st.session_state.chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.chat_history.append("assistant: " +
                                         st.session_state.models_dict["model_representation"]["model description"])
    st.session_state.detailed_chat_history.append("user: I have uploaded a Pyomo model.")
    st.session_state.detailed_chat_history.append("assistant: " +
                                                  st.session_state.models_dict["model_representation"][
                                                      "model description"])


chat_history_texts = '\n\n'.join(st.session_state.chat_history)
detailed_chat_history_texts = '\n\n'.join(st.session_state.detailed_chat_history)

st.sidebar.button("Process", on_click=process)
st.sidebar.button("Load JSON", on_click=load_json)


show_model_representation = st.sidebar.checkbox("Show Model Representation", False)
model_representation_placeholder = st.empty()
show_code = st.sidebar.checkbox("Show Code", False)
code_placeholder = st.empty()
show_tech_feedback = st.sidebar.checkbox("Show Technical Feedback", False)
tech_feedback_placeholder = st.empty()

st.sidebar.download_button(label="Export Chat History", data=chat_history_texts,
                           file_name='chat_history.txt', mime='text/plain')
st.sidebar.download_button(label="Export Detailed Chat History", data=detailed_chat_history_texts,
                           file_name='detailed_chat_history.txt', mime='text/plain')


st.sidebar.markdown("### Status")
status = st.sidebar.empty()

st.sidebar.markdown("### Round")
cur_round = st.sidebar.empty()

st.sidebar.markdown("### Agent")
agent_name = st.sidebar.empty()

st.sidebar.markdown("### Task")
task = st.sidebar.empty()


if show_model_representation:
    with model_representation_placeholder.container():
        st.json(st.session_state.models_dict["model_representation"])
else:
    model_representation_placeholder.empty()

if show_code:
    with code_placeholder.container():
        st.code(st.session_state.code)
else:
    code_placeholder.empty()

if show_tech_feedback:
    with tech_feedback_placeholder.container():
        for message in st.session_state.team_conversation:
            st.write(message['agent_name'] + ': ' + message['agent_response'])
            # if message['agent_name'] in ['Programmer', 'Operator', 'Syntax reminder', 'Explainer', 'Coordinator']:
            #     st.write(message['agent_name'] + ': ' + message['agent_response'])
else:
    tech_feedback_placeholder.empty()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#====function to take user input and get response based on previous conversation====

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Enter your query here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    updated_messages, team_conversation = OptiChat_workflow_exp(
        st.session_state,
        st.session_state.Coordinator,
        st.session_state.Engineer,
        st.session_state.Explainer,
        st.session_state.messages,
        st.session_state.models_dict
    )

    print('OptiChat_out:', updated_messages)

    if updated_messages:
        st.session_state.messages = updated_messages
    else:
        st.warning("No response received from OptiChat_workflow_exp.")

#=======================================================================
