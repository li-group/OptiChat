# Overview of OptiChat
A chatbot for diagnosing infeasible optimization problems. A GUI application powered by GPT LLM, equipped with custom tools, and aimed at helping unskilled operators and business people use plain English to troubleshoot infeasible optimization models.

# Installation
1. Install python3 and pip
2. Install python packages ```pip install -r requirements.txt```
3. Install Gurobi following the instructions  [here](https://support.gurobi.com/hc/en-us/articles/4534161999889). For windows without admin access, follow the instructions
[here](https://support.gurobi.com/hc/en-us/articles/360060996432-How-do-I-install-Gurobi-on-Windows-without-administrator-credentials-)
4. Apply for an OpenAI API key [here](https://platform.openai.com/). Add the key to your environment variables as ```OPENAI_API_KEY```
5. Run GUI.py in the src folder ```python GUI.py``` to use the chatbot

# Tutorial
Browse: Select your infeasible model (only support pyomo version .py file).

Process: Load your model and provide you with the first, preliminary report of your model.

Export: Export and save the chat history.


# Let's chat.
1. Get familiar with the model, you can ...

Ask general questions if you don't know optimization well.

Ask specific questions if you find any term or explanation unclear.

Ask for suggestions if you feel overwhelmed with information.


2. Let GPT troubleshoot, you only need to ...

Provide your request for changes in certain parameters that you believe may be relevant to addressing infeasibility.

You don't need to implement any code or read any code, just state something like: please help me change _____ and see if the model works or not now.


3. Understand the feasibility, you can ...

Ask follow-up questions once the model becomes feasible.

# Model Library:




