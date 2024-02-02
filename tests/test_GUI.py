import os, sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from src.Util import *
from src.GUI import *
from PySide6.QtWidgets import QApplication
import os
from time import sleep
import openai

app = QApplication([])

@pytest.fixture
def chatbot():
    return InfeasibleModelTroubleshooter()


def get_completion(summary, gpt_model="gpt-4"):
    messages = []
    system_prompt = {
        "role": "system",
        "content": f"""
        The user is an mixed integer programming and linear programming expert. He has an optimization model written in Pyomo
        using which we solves the model and answers questions about the model. You are his loyal assistant and he will tip you
        $100 for every good question that you ask him about the model. Don't worry, he is a very generous person and will
        provide you with the summary of the model. 
        
        There are 5 categories of good questions that you as an assistant should ask the user. They are:

        1. Infeasibility troubleshooting. What this means is that the user will check if any of the constraints
        of the optimization model are making it infeasible, and will identify what are the parameters that are involved
        in these constraints. He adds slack variables to these parameters and try to resolve the model. Example queries of this kind
        are:
            
            a) "Why can't I find a solution to my production planning problem even though I've listed out all my constraints? Are any of them conflicting with each other?"
            b) "I've set up an optimal staffing schedule to minimize costs, but the solver says there's no feasible solution. Can we figure out which staffing requirement is causing the issue?"
            c) "I'm trying to optimize the routing for my delivery trucks, but it's not working out. Could there be any route or time constraints that are impossible to meet together?"
            d) "My inventory optimization model was working fine last month. This month, I can't get a solution. What might have changed in the demand or supply constraints that's causing this?"
            e) "I've modeled a diet plan to minimize costs while meeting all nutrient requirements. However, I can't find a feasible diet. Are there any nutrient requirements that are contradicting each other or impossible to meet with the given food items?"

        2.Sensitivity analysis. Sensitivity analysis in linear programming (LP) refers to the study of how changes in the coefficients and
        parameters of a linear program impact the optimal solution. In business and engineering, this analysis is crucial 
        because it provides insight into the robustness and reliability of an optimal solution under various scenarios.
        Some example queries of this kind are:

            a) "If we can get an extra hour of machine time each day, how much more profit can we expect? Is it worth paying overtime for the workers?"
            b) "How much more would our transportation costs increase if fuel prices went up by 10%? Should we consider negotiating long-term fuel contracts now?"
            c) "Suppose there's a slight decrease in the yield of our main crop due to unexpected weather changes. How would this affect our yearly revenue? Should we consider diversifying our crops next season?"
            d) "If we allocate an additional $10,000 to our marketing budget, how much more revenue can we expect? Is it a better return on investment than, say, investing in product development?"
            e) "How would extending our customer service hours by two hours every day affect our monthly operating costs? And if we did extend the hours, would it significantly improve our customer satisfaction ratings?"
        
        3. User already has the information on what constraints of the model are causing it to be infeasible/feasible. He also knows what are the parameters of the model that are in these constriants. He also knows the background story of the optimization model
        and the real-world meaning of the model constraints, parameters and variables. With all this information, he will just answer the queries without re-solving/troubleshooting the model.
        Example queries of this kind are:

            a) "What are the constraints that are causing my model to be not feasible?"
            b) "what are the constraints that are making my model feasbile?"
            c) "What is my model about?"
            d) "What physical quantities are making the model infeasible?"
            e) "What are the parameters that I need to change to make the model feasible?"
            f) "Which staffing requirements make my work schedule optimization model infeasible?"
            g) "Explain the complete story of my model."

        4. The user has access to the pyomo code of the model (which has detailed doc string and comments) and also has a concise summary of the
        model parameters, whether they are indexed, and if indexed then their dimension and all the indices etc as a json object. So he has information that can be obtained only
        by looking up the pyomo code file and with the knowledge of the python programming language. But however, he does not know their physical meaning/the complete background story like the category "3." above.
        Example queries of this kind are:

            a) "What are the indexed parameters present in the model?"
            b) "If ('a', 1) a valid index for demand?"
            c) "What are the indices of the parameter `ship_class`?"
            d) "How many different kinds of ships are there? What are their capacities?"
            e) "What are the different kinds of ships we have?"
            f) "How many men are present in Surat?"
        
        5. The user has access to the optimization model which is written in pyomo. You will ask him questions about the optimal value of the objective of the model,
        or the optimal values of different variables present in the model. You can assume that the model has already been solved, so these queries are just about the optimal solution of the model.
        Example queries of this kind are:
            a) "What is the optimal cost in my problem?"
            b) "What are the optimal values of the variables in my problem?"
            c) "Which variable has the highest value in the optimal solution?"
            d) "What is the optimal value of the objective function?"
            e) "What is the optimal value of the variable `x`?"
            f) "How many generators should I have in my power plant in order to have the maximum profit?"
        
        GENERATE 5 QUESTIONS THAT YOU CAN ASK THE USER. GENERATE ONLY WHAT IS ASKED. DO NOT GENERATE ANY EXTRA TEXT.
        """
    }
    messages.append({"role": "user", "content": f"Hey my friend. Here is the model summary for you: {summary}. I will tip you $500 in total if you help me. Generate 5 good questions that you can ask me."})
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    response = client.chat.completions.create(model=gpt_model,
    messages = messages,
    temperature=0)
    return response.choices[0].message.content



def test_sentivity_GUI_manual(chatbot):
    file_names = ["./Pyomo_Model_Lib/feasibleproblems/thai_v0.py", "./Pyomo_Model_Lib/feasibleproblems/bid_v0.py", "./Pyomo_Model_Lib/feasibleproblems/magic_v0.py"]

    for file_name in file_names:
        chatbot.py_path = os.path.abspath(file_name)
        chatbot.process_file()

        chatbot.process_thread.run()
        chatbot.process_thread.wait()
        chatbot.process_thread.quit()

        chatbot.txt_in.setPlainText("What is the model about?")
        chatbot.enter()
        chatbot.chat_thread.run()
        chatbot.chat_thread.wait()
        chatbot.chat_thread.quit()

        model_params_info = chatbot.chat_thread.model_info

        print(model_params_info)

        for param in model_params_info:
            chatbot.txt_in.setPlainText(f"What is the rate of change of optimal cost with {param}?")
            chatbot.enter()
            chatbot.chat_thread.run()
            chatbot.chat_thread.wait()
            chatbot.chat_thread.quit()

            chatbot.txt_in.setPlainText(f"What are the indices of the parameter {param}?")
            chatbot.enter()
            chatbot.chat_thread.run()
            chatbot.chat_thread.wait()
            chatbot.chat_thread.quit()


            chatbot.txt_in.setPlainText(f"How much will the total cost increase if I increase my {param} by 10 percentage?")
            chatbot.enter()
            chatbot.chat_thread.run()
            chatbot.chat_thread.wait()
            chatbot.chat_thread.quit()

    
    assert True


@pytest.mark.parametrize("file_name", ["./Pyomo_Model_Lib/feasibleproblems/thai_v0.py"]) #"./Pyomo_Model_Lib/feasibleproblems/bid_v0.py", "./Pyomo_Model_Lib/feasibleproblems/magic_v0.py"])
def test_GUI(chatbot, file_name):
    # file_name = "./Pyomo_Model_Lib/feasibleproblems/thai_v0.py"
    chatbot.py_path = os.path.abspath(file_name)
    chatbot.process_file()

    print(chatbot.chatbot_messages)
    print(chatbot.process_thread.isFinished())
    chatbot.process_thread.run()
    chatbot.process_thread.wait()
    chatbot.process_thread.quit()

    print('SUMMARY', chatbot.summary)

    print("=============================================" * 10)

    print("INFEASIBILITY REPORT", chatbot.infeasibility_report)

    print("=============================================" * 10)

    response = get_completion(chatbot.summary + '\n' + chatbot.infeasibility_report)

    print(response.split("\n"))

    for question in response.split("\n"):
        chatbot.txt_in.setPlainText(question)
        chatbot.enter()
        chatbot.chat_thread.run()
        chatbot.chat_thread.wait()
        chatbot.chat_thread.quit()
        
    # chatbot.txt_in.setPlainText("What is this model about?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What are the parameters present in this model?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What are the variables present in this model?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What is the optimal cost for this complex operation?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What are the different ship classes present in the model?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What are the different voyages present in the model?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("What are the ports present in the model?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("For minimum cost of operation, how many men are transported from chumphon via voyage v-01 with small ships?")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()

    # chatbot.txt_in.setPlainText("Give me the optimal values for all the variables present in my model")
    # chatbot.enter()

    # chatbot.chat_thread.run()
    # chatbot.chat_thread.wait()
    # chatbot.chat_thread.quit()


    # # print(chatbot.chatbot_messages)
    # # print(chatbot.chat_thread.isFinished())

    # # print(chatbot.chat_thread.ai_message)
    # # print("SUMMARY", chatbot.summary)
    assert True