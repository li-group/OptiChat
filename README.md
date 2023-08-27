# OptiChat: Diagnosing Infeasible Optimization Problems Using Large Language Models
Welcome to the [OptiChat paper](https://arxiv.org/abs/2308.12923) page. OptiChat is a chatbot powered by GPT LLM, equipped with custom tools, and aimed at helping unskilled operators and business people use plain English to troubleshoot infeasible optimization models.

We aspire for this project to provide some valuable insights into **linking explanation/troubleshooting methodologies in optimization with LLM**, thereby facilitating the identification of the infeasibility sources and streamlining the troubleshooting process by non-experts 🚀. 

If you found this work useful, please cite this [preprint](https://arxiv.org/abs/2308.12923) as:
```bibtex
@misc{chen2023diagnosing,
    title={Diagnosing Infeasible Optimization Problems Using Large Language Models},
    author={Hao Chen and Gonzalo E. Constante-Flores and Can Li},
    year={2023},
    eprint={2308.12923},
    archivePrefix={arXiv},
    primaryClass={cs.HC}
}
```

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Chat Example](#chat-example)
- [Model Library](#model-library)
- [Build Your Own (Infeasibile) Model and Test it](#build-your-own-model-and-test-it)
- [Citation](#citation)

# Overview
<a name="overview"></a>
## Motivation
Mathematical optimization models are extensively utilized to solve decision-making problems across various fields,, but there may exist no solution that satisfies all the constraints. This situation often arises in practice and stands as key obstacles in the implementation by industrial operators. It becomes important that anyone without much knowledge of mathematical programming can understand the infeasibility and relate it to potential strategies in the real world, rather than involving an expert in the code and digging the infeasibility sources. 

Please read our [OptiChat paper](https://arxiv.org/abs/2308.12923) for more motivation and details about how the OptiChat works.

## Scope
Model: OptiChat supports infeasible optimization model written in [Pyomo](http://www.pyomo.org).

Troubleshoot: OptiChat supports request for the changes in terms of parameters (and constraints) and automates the identification of strategy that requires the minimal effort to make the model feasible. 

In the future work, we will add more functionalities such as specifying changes for particular indices.

# Installation
<a name="installation"></a>
1. Install python3 and pip
2. Install python packages ```pip install -r requirements.txt```
3. Install Gurobi following the instructions in the youtube videos  [here](https://support.gurobi.com/hc/en-us/articles/4534161999889). For windows without admin access, follow the instructions
[here](https://support.gurobi.com/hc/en-us/articles/360060996432-How-do-I-install-Gurobi-on-Windows-without-administrator-credentials-)
4. Apply for an OpenAI API key [here](https://platform.openai.com/).  Add the key to your environment variables as ```OPENAI_API_KEY```
5. To check whether the installation of gurobi and GPT is successful, at the root directory, run ```pytest tests/```. If the test passes, you are good to go. 
6. Run GUI.py in the src folder ```python GUI.py``` to use the chatbot

# Tutorial
<a name="tutorial"></a>
1. Browse: Select your infeasible model (only support pyomo version .py file). There are a number of infeasible models in **Pyomo_Model_Lib** folder for you to test.

2. Process: Load your model and provide you with the first, preliminary report of your model. For best performance, use the gpt-4 model. For faster speed, use gpt-3.5-turbo model. **This step usually takes a few minutes**, please wait until the "Loading the model..." prompt becomes "GPT responded."

3. Chat: Diagnose and troubleshoot through a natural-language conversation ([Chat Example](#chat-example))

4. Export: Export and save the chat history.


## Chat Example
<a name="chat-example"></a>
<p align="center">
<img src="images/Chatbot_eg_highPPI.png" alt="drawing" width="700"/>
</p>

1. Get familiar with the model, you can ...

Ask general questions if you don't know optimization well.

Ask specific questions if you find any term or explanation unclear.

Ask for suggestions if you feel overwhelmed with information.


2. Let GPT troubleshoot, you only need to ...

Provide your request for changes in certain parameters that you believe may be relevant for addressing infeasibility.

You don't need to implement any code or read any code, just state something like: **please help me change _____ and see if the model works or not now**.


3. Understand the feasibility, you can ...

Ask follow-up questions once the model becomes feasible.
Provide additional requests for changes in other parameters that you find relevant.

# Model Library:
<a name="model-library"></a>
The model libary is located in the **Pyomo_Model_Lib** folder.

# Build Your Own (Infeasibile) Model and Test it:
<a name="build-your-own-model-and-test-it"></a>
At the current stage, OptiChat only supports optimization models written in Pyomo. A typical Pyomo model example is given as follows. 
**Please remember to set parameters "mutable=True" unless you are entirely certain that a parameter cannot be altered in any manner (eg. task duration in scheduling). This is because pyomo only has the mutable parameters in symbolic form, which is essential for GPT to work well.** Try to add some comments for the parameters and variables.
```python
from pyomo.environ import *

model = ConcreteModel()

# Set
model.t = Set(initialize=[1, 2, 3, 4, 5])  # Time periods (weeks)

# Parameters
# Worker productivity (units per worker)
model.rho = Param(initialize=8, mutable=True)
# Trainer capability (workers per trainer)
model.alpha = Param(initialize=6, mutable=True)
# Worker wages ($ per week per worker)
model.wage = Param(initialize=100, mutable=True)
# Initial stock of goods (units)
model.si = Param(model.t, initialize={1: 10}, mutable=True)  
# Initial number of workers (workers)
model.wi = Param(model.t, initialize={1: 20}, mutable=True)  
# Salary on firing ($)
model.sf = Param(model.t, initialize={5: 100}, default=0, mutable=True)  
# Demand schedule (units)
model.d = Param(model.t, initialize={1: 100, 2: 600, 3: 300, 4: 400, 5: 200}, mutable=True)  

# Variables
model.p = Var(model.t, within=NonNegativeReals)  # Production level in period t (units)
model.s = Var(model.t, within=NonNegativeReals)  # Goods stored in period t (units)
model.w = Var(model.t, within=NonNegativeReals)  # Total potential productive workers (workers)
model.h = Var(model.t, within=NonNegativeReals)  # Workers hired (workers)
model.f = Var(model.t, within=NonNegativeReals)  # Workers fired (workers)

# Constraints
def cb_rule(model, t):
    if t == 1:
        return model.s[t] == model.si[t] + model.p[t] - model.d[t]
    else:
        return model.s[t] == model.s[t - 1] + model.p[t] - model.d[t]
model.cb = Constraint(model.t, rule=cb_rule)  # Commodity balance constraint

def wb_rule(model, t):
    if t == 1:
        return model.w[t] == model.wi[t] - model.f[t] + model.h[t]
    else:
        return model.w[t] == model.w[t - 1] - model.f[t] + model.h[t]
model.wb = Constraint(model.t, rule=wb_rule)  # Worker balance - between periods constraint

def wd_rule(model, t):
    return model.w[t] >= model.p[t] / model.rho + (1 + 1 / model.alpha) * model.h[t]
model.wd = Constraint(model.t, rule=wd_rule)  # Worker balance - job differentiation constraint

# Objective
model.obj = Objective(expr=sum(10 * model.s[t] + (model.wage + model.sf[t]) * model.w[t] for t in model.t), sense=minimize)
```

# Citation
<a name="citation"></a>
Cite us ❤️
```bibtex
@misc{chen2023diagnosing,
    title={Diagnosing Infeasible Optimization Problems Using Large Language Models},
    author={Hao Chen and Gonzalo E. Constante-Flores and Can Li},
    year={2023},
    eprint={2308.12923},
    archivePrefix={arXiv},
    primaryClass={cs.HC}
}
```

