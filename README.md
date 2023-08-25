# Overview of OptiChat
A chatbot for diagnosing infeasible optimization problems. A GUI application powered by GPT LLM, equipped with custom tools, and aimed at helping unskilled operators and business people use plain English to troubleshoot infeasible optimization models.

## Table of Contents
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Chat Example](#chat-example)
- [Model Library](#model-library)
- [Build Your Own (Infeasibile) Model and Test it](#build-your-own-model-and-test-it)


# Installation
<a name="installation"></a>
1. Install python3 and pip
2. Install python packages ```pip install -r requirements.txt```
3. Install Gurobi following the instructions in the youtube videos  [here](https://support.gurobi.com/hc/en-us/articles/4534161999889). For windows without admin access, follow the instructions
[here](https://support.gurobi.com/hc/en-us/articles/360060996432-How-do-I-install-Gurobi-on-Windows-without-administrator-credentials-)
4. Apply for an OpenAI API key [here](https://platform.openai.com/). Add the key to your environment variables as ```OPENAI_API_KEY```
5. To check whether the installation of gurobi and GPT is successful, at the root directory, run ```pytest tests/```. If the test passes, you are good to go. 
6. Run GUI.py in the src folder ```python GUI.py``` to use the chatbot

# Tutorial
<a name="tutorial"></a>
Browse: Select your infeasible model (only support pyomo version .py file). There are a number of infeasible models in Pyomo_Model_Lib folder for you to test. 

Process: Load your model and provide you with the first, preliminary report of your model. This step usually takes a few minutes, please wait until the "Loading the model..." prompt becomes "GPT responded."

Export: Export and save the chat history.


# Chat Example
<a name="chat-example"></a>
![An illustrative conversation](https://github.com/li-group/OptiChat/blob/main/images/Chatbot_eg.png)

1. Get familiar with the model, you can ...

Ask general questions if you don't know optimization well.

Ask specific questions if you find any term or explanation unclear.

Ask for suggestions if you feel overwhelmed with information.


2. Let GPT troubleshoot, you only need to ...

Provide your request for changes in certain parameters that you believe may be relevant to addressing infeasibility.

You don't need to implement any code or read any code, just state something like: please help me change _____ and see if the model works or not now.


3. Understand the feasibility, you can ...

Ask follow-up questions once the model becomes feasible.
Provide additional requests for changes in other parameters that you find relevant.

# Model Library:
<a name="model-library"></a>
The model libary is located in the Pyomo_Model_Lib folder.

# Build Your Own (Infeasibile) Model and Test it:
<a name="build-your-own-model-and-test-it"></a>
At the current stage, OptiChat only supports optimization models written in Pyomo. A typical Pyomo model example is given as follows: 





