from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent, ParallelAgent, Agent
from google.adk.tools.agent_tool import AgentTool
from optichat.llm import *
from optichat.config.constants import *
from optichat.sub_agents.root.prompt import *
from optichat.sub_agents.expert.agent import create_expert_agent
from optichat.tools.callback_tool import (initialize_session, initialize_query)



def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}
    


def create_root_agent(workflow="default"):
    if workflow == "default":
        expert_agent = create_expert_agent(prompt_version=1, tools_version=1)
        root_agent = Agent(name="root_agent",
                           model=gpt_5_nano,
                           tools=[AgentTool(expert_agent)],
                           description="first point of contact for all user queries",
                           instruction=ROOT_AGENT_PROMPT,
                           output_key=OUTPUT_KEY_ROOT_AGENT,
                           before_agent_callback=initialize_session,
                           before_model_callback=initialize_query)
        return root_agent
    else:
        raise NotImplementedError(f"Workflow '{workflow}' is not implemented.")