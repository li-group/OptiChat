import copy
import time
from typing import Dict, Optional, Union, List
from openai import Client, OpenAI
from prompts import get_prompts
from internal_tools import feasibility_restoration, sensitivity_analysis, components_retrival, evaluate_modification
from internal_tools import syntax_guidance, fnArgsDecoder
from extractor import extract_component_descriptions, insert_code, run_with_exec
import json
import re
#import streamlit as st


class Agent:
    def __init__(self, name, description, client, llm="gpt-4-turbo-preview", **kwargs):
        self.name = name
        self.description = description
        self.client = client
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.llm = llm

        self.function_names = kwargs.get('function_names', None)
        self.tools = kwargs.get('tools', None)
        self.multiple_tools = kwargs.get('multiple_tools', None)
        self.single_tools = kwargs.get('single_tools', None)
        self.none_tools = kwargs.get('none_tools', None)
        self.all_tools = kwargs.get('all_tools', None)
        self.tool_choice = kwargs.get('tool_choice', None)
        self.syntax_guidance_tool = kwargs.get('syntax_guidance_tool', None)

        self.team_conversation_filename = './logs/team_conversation.txt'
        self.chat_history_filename = './logs/detailed_chat_history.txt'

    def llm_call(self, prompt: Optional[str] = None, messages: Optional[List] = None,
                 seed: int = 10, stream: bool = False) -> str:
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if not prompt is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        # print("=" * 10)
        # print(f'llm_call is called, the following messages are sent to the llm: ')
        # for message in messages:
        #     print(f'{message["role"]}: {message["content"]}')
        # print("=" * 10)

        if type(self.client) in [OpenAI, Client]:
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                seed=seed,
                stream=stream,
            )

            if stream:
                return completion
            else:
                content = completion.choices[0].message.content
                return content

    @staticmethod
    def generate_pseudo_messages(messages: List[Dict], team_conversation: List[Dict],
                                 new_prompt: str) -> List[Dict]:
        pseudo_messages = copy.deepcopy(messages)
        if team_conversation:
            for message in team_conversation:
                if message["agent_name"] in ['Syntax reminder', 'Code reminder']:
                    pseudo_messages.append({"role": "system",
                                            "content": f'{message["agent_name"]}: \n\n' +
                                                       message["agent_response"]})
                else:
                    pseudo_messages.append({"role": "assistant",
                                            "content": f'I am {message["agent_name"]} in Assistant Team. \n\n' +
                                                       message["agent_response"]})
        pseudo_messages.append({"role": "user", "content": new_prompt})
        return pseudo_messages

    def save_team_conversation(self, team_conversation):
        with open(self.team_conversation_filename, 'a') as f:
            for message in team_conversation:
                f.write(f"{message['agent_name']}: {message['agent_response']}\n\n")

    def print_in_and_out(self, prompt, llm_response, agent_name=None):
        if agent_name is None:
            agent_name = self.name
        print("=" * 5 + agent_name + "=" * 5)
        print('-' * 5 + 'prompt:' + '-' * 5)
        print(prompt)
        print('-' * 5 + 'llm_response:' + '-' * 5)
        print(llm_response)

    def llm_call_exp(self, prompt: Optional[str] = None, messages: Optional[List] = None,
                     seed: int = 10, temperature: float = 0.1,
                     json_mode: bool = False,
                     stream: bool = False,) -> str:
        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if not prompt is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        if json_mode:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        if type(self.client) in [OpenAI, Client]:
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                seed=seed,
                temperature=temperature,
                response_format=response_format,
                stream=stream,
                )

            if stream:
                return completion
            else:
                content = completion.choices[0].message.content
                return content


class Interpreter(Agent):
    def __init__(self, client: Client, **kwargs):
        super().__init__(
            name="Interpreter",
            description="This is an operations research agent that is an expert in interpreting optimization models and codes to non-experts.",
            client=client,
            **kwargs,
        )

        self._init_prompt_template()

    def _init_prompt_template(self):
        self.interpretation_prompt_template = get_prompts("model_interpretation_prompt")
        self.need2describe_prompt_template = get_prompts("need2describe_prompt")
        self.interpretation_json_template = get_prompts("model_interpretation_json")

        self.illustration_prompt_template = get_prompts("model_illustration_prompt")
        self.inference_prompt_template = get_prompts("model_inference_prompt")

    def _cat(self, cat_need2describe, component_names, component_type):
        return cat_need2describe + self.need2describe_prompt_template.format(component_type=component_type,
                                                                             component_names=component_names)

    def _cut(self, component_type):
        if component_type in self.interpretation_json_template["components"]:
            del self.interpretation_json_template["components"][component_type]

    def generate_interpretation(self, models_dict: Dict, code: str, model_name="model_1"):
        task_complete = False
        cnt = 3
        while not task_complete and cnt > 0:
            self._init_prompt_template()
            cat_need2describe_prompt = ""
            need2describe = {}
            for component_type in ['sets', 'parameters', 'variables', 'constraints', 'objective']:
                need2describe[component_type] = []
                for key, value in models_dict[model_name]["components"][component_type].items():
                    if value.get('description') in ['None', None]:
                        need2describe[component_type].append(key)
                # if there are components that haven't been described, add them to the prompt
                if len(need2describe[component_type]) > 0:
                    cat_need2describe_prompt = self._cat(cat_need2describe_prompt,
                                                         need2describe[component_type], component_type)
                else:
                    self._cut(component_type)
                # print('===' * 10)
                # print('cat_need2describe_prompt:', cat_need2describe_prompt)
                # print(f'interpretation_json_template: {self.interpretation_json_template}')

            if len(cat_need2describe_prompt) > 0:
                model_interpretation_json = json.dumps(self.interpretation_json_template, indent=4)
                # create complete prompt with components that haven't been described only
                prompt = self.interpretation_prompt_template.format(code=code,
                                                                    cat_need2describe_prompt=cat_need2describe_prompt,
                                                                    model_interpretation_json=model_interpretation_json)
            else:
                # if all the components in all the component types have been described, then no need to call interpreter
                return models_dict

            cnt -= 1
            try:
                interpretation_json = self.llm_call(prompt=prompt, seed=cnt, stream=False)
                print("=" * 10)
                print(f'generate_interpretation... cnt left = {cnt}/3')
                print(interpretation_json)
                print("=" * 10)
                output = interpretation_json
                # delete until the first '```json'
                if "```json" in output:
                    output = output[output.find("```json") + 7:]
                    output = output[: output.rfind("```")]

                start = output.find("{")
                end = output.rfind("}")
                output = output[start:end + 1]

                update = json.loads(output)

                task_complete = True  # mark as complete first, if any component incorrect, mark as incomplete
                for key in update["components"]:
                    print(f'Interpreting {key}')
                    for component in update["components"][key]:
                        print(f'component: {component}')
                        # update models_dict with the new descriptions if format is correct,
                        # next time less components will be included in the prompt
                        if ('name' in component) and ('description' in component):
                            models_dict[model_name]["components"][key][component["name"]]["description"] = component[
                                "description"]
                        else:
                            print(f'Invalid component format marked!, {component}')
                            task_complete = False

            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print("=" * 10)
                print(f'generate_interpretation error... cnt left = {cnt}/3')
                print(e)
                print("=" * 10)
                print(f'generate_interpretation prompt that caused the error: ')
                print(prompt)
                print("=" * 10)
                print(interpretation_json)
                print("=" * 10)
                print(
                    f"Invalid json format!\n{e}\n Try again ..."
                )
        if cnt == 0:
            raise Exception("Invalid json format, Failed 3 times!")
        return models_dict

    def generate_illustration(self, model_representation: Dict):
        prompt = self.illustration_prompt_template.format(
            json_representation=model_representation)
        print("=" * 10)
        print(f'generate_illustration... ')
        print("=" * 10)
        stream = self.llm_call(prompt=prompt, stream=True)
        return stream

    def generate_inference(self, model_representation: Dict):
        def split_representation(representation):
            # just split session_state.models_dict["model_representation"] into two parts
            reduced_json_representation = copy.deepcopy(representation)
            del reduced_json_representation["iis"]
            del reduced_json_representation["iis_description"]
            return representation["iis_description"], reduced_json_representation

        iis_info, reduced_model_representation = split_representation(model_representation)
        prompt = self.inference_prompt_template.format(
            iis_info=iis_info,
            json_representation=reduced_model_representation)
        print("=" * 10)
        print(f'generate_inference... ')
        print("=" * 10)
        stream = self.llm_call(prompt=prompt, stream=True)
        return stream

    def generate_interpretation_exp(self, args, models_dict: Dict, code: str, model_name="model_1"):
        task_complete = False
        cnt = 3
        while not task_complete and cnt > 0:
            self._init_prompt_template()
            cat_need2describe_prompt = ""
            need2describe = {}
            for component_type in ['sets', 'parameters', 'variables', 'constraints', 'objective']:
                need2describe[component_type] = []
                for key, value in models_dict[model_name]["components"][component_type].items():
                    if value.get('description') in ['None', None]:
                        need2describe[component_type].append(key)
                # if there are components that haven't been described, add them to the prompt
                if len(need2describe[component_type]) > 0:
                    cat_need2describe_prompt = self._cat(cat_need2describe_prompt,
                                                         need2describe[component_type], component_type)
                else:
                    self._cut(component_type)

            if len(cat_need2describe_prompt) > 0:
                model_interpretation_json = json.dumps(self.interpretation_json_template, indent=4)
                # create complete prompt with components that haven't been described only
                prompt = self.interpretation_prompt_template.format(code=code,
                                                                    cat_need2describe_prompt=cat_need2describe_prompt,
                                                                    model_interpretation_json=model_interpretation_json)
            else:
                # if all the components in all the component types have been described, then no need to call interpreter
                task_complete = True
                return models_dict, cnt, task_complete

            cnt -= 1
            try:
                interpretation_json = self.llm_call_exp(prompt=prompt, seed=cnt,
                                                        temperature=args.temperature,
                                                        json_mode=args.json_mode, stream=False)
                print("=" * 10)
                print(f'generate_interpretation... cnt left = {cnt}/3')
                print(interpretation_json)
                print("=" * 10)
                output = interpretation_json

                # print("=" * 10 + 'debug: for testing json mode only' + "=" * 10)
                # print(output)
                # print("=" * 10)

                # delete until the first '```json'
                if "```json" in output:
                    output = output[output.find("```json") + 7:]
                    output = output[: output.rfind("```")]

                start = output.find("{")
                end = output.rfind("}")
                output = output[start:end + 1]

                update = json.loads(output)

                task_complete = True  # mark as complete first, if any component incorrect, mark as incomplete
                for key in update["components"]:
                    print(f'Interpreting {key}')
                    for component in update["components"][key]:
                        print(f'component: {component}')
                        # update models_dict with the new descriptions if format is correct,
                        # next time less components will be included in the prompt
                        if ('name' in component) and ('description' in component):
                            models_dict[model_name]["components"][key][component["name"]]["description"] = component[
                                "description"]
                        else:
                            print(f'Invalid component format marked!, {component}')
                            task_complete = False

            except Exception as e:
                import traceback

                print(traceback.format_exc())
                print("=" * 10)
                print(f'generate_interpretation error... cnt left = {cnt}/3')
                print(e)
                print("=" * 10)
                print(
                    f"Invalid json format!\n{e}\n Try again ..."
                )
        if cnt == 0:
            return models_dict, cnt, task_complete
        return models_dict, cnt, task_complete

    def generate_illustration_exp(self, args, model_representation: Dict):
        prompt = self.illustration_prompt_template.format(
            json_representation=model_representation)
        print("=" * 10)
        print(f'generate_illustration... ')
        print("=" * 10)
        stream_or_completion = self.llm_call_exp(prompt=prompt, temperature=args.temperature,
                                                 stream=args.illustration_stream)
        return stream_or_completion

    def generate_inference_exp(self, args, model_representation: Dict):
        def split_representation(representation):
            # just split session_state.models_dict["model_representation"] into two parts
            reduced_json_representation = copy.deepcopy(representation)
            del reduced_json_representation["iis"]
            del reduced_json_representation["iis_description"]
            return representation["iis_description"], reduced_json_representation

        iis_info, reduced_model_representation = split_representation(model_representation)
        prompt = self.inference_prompt_template.format(
            iis_info=iis_info,
            json_representation=reduced_model_representation)
        print("=" * 10)
        print(f'generate_inference... ')
        print("=" * 10)
        stream_or_completion = self.llm_call_exp(prompt=prompt, temperature=args.temperature,
                                                 stream=args.inference_stream)
        return stream_or_completion


class Coordinator(Agent):
    def __init__(
        self, client: Client, agents: [Agent], max_rounds: int = 5, **kwargs
    ):
        super().__init__(
            name="Coordinator",
            description="This is a coordinator agent that chooses which agent to work on the problem next and organizes "
                        "the conversation within its team. ",
            client=client,
            **kwargs,
        )

        self.agents = agents
        self.max_rounds = max_rounds

        self.coordination_time = 0
        self.coordinator_success = False
        self._init_cnt()
        self._init_prompt_template()

    def _init_cnt(self):
        self.coordinator_cnt = 3
        self.coordinator_success = False

    def _init_prompt_template(self):
        self.prompt_template = get_prompts("coordinator_prompt")
        self.agents_list = "".join(
            [
                "-" + agent.name + ": " + agent.description + "\n"
                for agent in self.agents
            ]
        )

    def generate_decision(self, messages, team_conversation, agent_name, task):
        status = 'In Progress'

        coordinate_prompt = self.prompt_template.format(agents=self.agents_list)
        pseudo_messages = self.generate_pseudo_messages(messages, team_conversation, coordinate_prompt)

        cnt = 3
        while cnt > 0:
            try:
                response = self.llm_call(messages=pseudo_messages, seed=cnt)
                decision = response.strip()
                if "```json" in decision:
                    decision = decision.split("```json")[1].split("```")[0]
                decision = decision.replace("\\", "")

                self.print_in_and_out(coordinate_prompt, response)
                print('Decision:', decision)

                decision = json.loads(decision)

                if team_conversation:
                    # safeguard to prevent the coordinator from calling the agent
                    # after the user's query has been answered by explainer
                    if team_conversation[-1]["agent_name"] == "Explainer":
                        status = 'Completed'
                        OptiChat_out = team_conversation[-1]['agent_response']
                        if "DONE" in decision.values():
                            print("DONE, the user's query is answered.")
                        else:
                            print("DONE, the user's query is answered, though the coordinator did not output 'DONE'.")
                        return status, OptiChat_out
                    if "DONE" in decision.values() and team_conversation[-1]["agent_name"] == "Engineer":
                        decision = {'agent_name': 'Explainer', 'task': 'explain the technical feedback'}

                else:
                    # the first round of the conversation
                    if "DONE" in decision.values():
                        # sometimes user does not ask a question (e.g. saying 'thank you')
                        # and coordinator considers no query there and outputs 'DONE' directly
                        decision = {'agent_name': 'Explainer', 'task': 'respond to the user'}

                agent_name.text(decision["agent_name"])
                task.text(decision["task"])

                return status, decision

            except Exception as e:
                print(e)
                cnt -= 1
                print("Invalid decision. Trying again ...")

                task.text(f'distribution failed ({cnt}/3)')

                if cnt == 0:
                    import traceback
                    err = traceback.format_exc()
                    print(err)

                    status = 'Terminated'
                    OptiChat_out = "LLM failed to assign tasks to experts! \n" + "Error: " + err + "\n"

                    return status, OptiChat_out

    def generate_decision_exp(self, args, messages, team_conversation):
        self._init_cnt()
        coordinate_prompt = self.prompt_template.format(agents=self.agents_list)
        while self.coordinator_cnt > 0:
            # messages will only be updated outside the loop (in the OptiChat workflow fn)
            # team_conversation will be updated inside the loop (in the Engineer and Explainer fns)
            pseudo_messages = self.generate_pseudo_messages(messages, team_conversation, coordinate_prompt)
            try:
                # in current design, if coordinator has assigned the task once,
                # actually there will be no need to call llm to generate the decision again
                if team_conversation:
                    decision = {'agent_name': 'Explainer', 'task': 'explain the technical feedback'}
                    # last_agent = team_conversation[-1]["agent_name"]
                    # if last_agent == "Engineer":
                    #     decision = {'agent_name': 'Explainer', 'task': 'explain the technical feedback'}
                else:
                    response = self.llm_call_exp(messages=pseudo_messages, seed=self.coordinator_cnt,
                                                 temperature=args.temperature,
                                                 json_mode=args.json_mode, stream=False)
                    decision = response.strip()
                    if "```json" in decision:
                        decision = decision.split("```json")[1].split("```")[0]
                    decision = decision.replace("\\", "")

                    print('Decision:', decision)

                    decision = json.loads(decision)
                    assert "agent_name" in decision
                    assert decision["agent_name"] in [agent.name for agent in self.agents]
                    assert "task" in decision

                    # in the first round of the conversation
                    # sometimes user does not ask a question (e.g. saying 'thank you')
                    # and coordinator considers no query there and outputs 'DONE' directly
                    if "DONE" in decision.values():
                        decision = {'agent_name': 'Explainer', 'task': 'respond to the user'}

                self.coordinator_success = True
                return decision

            except Exception as e:
                print(e)
                self.coordinator_cnt -= 1
                print("Invalid decision. Trying again ...")

                if self.coordinator_cnt == 0:
                    import traceback
                    err = traceback.format_exc()
                    print(err)
                    return None


class Explainer(Agent):
    def __init__(
        self, client: Client, max_rounds: int = 5, **kwargs
    ):
        super().__init__(
            name="Explainer",
            description="This is an explainer agent whose task is to either (1) directly answer user queries if the questions can be analyzed through natural language only, or (2) summarize the technical feedback obtained from engineers to answer user queries",
            client=client,
            **kwargs,
        )
        self.explanation_time = 0
        self._init_prompt_template()

    def _init_prompt_template(self):
        self.prompt_template = get_prompts("explainer_prompt")

    def generate_explanation_exp(self, args, messages, team_conversation):
        prompt = self.prompt_template  # nothing to format here
        pseudo_messages = self.generate_pseudo_messages(messages, team_conversation, prompt)

        stream_or_completion = self.llm_call_exp(messages=pseudo_messages, temperature=args.temperature,
                                                 stream=args.explanation_stream)
        return stream_or_completion


class Engineer(Agent):
    def __init__(self, client: Client, **kwargs):
        super().__init__(
            name="Engineer",
            description="This is an engineer agent whose task is to execute tools and functions when user's query requires an interaction with optimization model. The engineer agent provides technical feedback instead of natural-language explanations."
                        "Note that some ‘why’ questions are better answered with technical feedback."
                        "These questions often involve scenarios that differ from the current model.",
            client=client,
            **kwargs,
        )

        self.pattern = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"

        self._init_prompt_template()

        self.syntax_time = 0
        self.programing_time = 0
        self.evaluation_time = 0

        self.programmer_cnt = 3
        self.evaluator_cnt = 3

        self.syntax_success = False
        self.operator_success = False
        self.programmer_success = False
        self.evaluator_success = False

        self.unparsed_queried_components = None
        self.queried_components = None
        self.queried_model = None
        self.queried_function = None

    def _init_prompt_template(self):
        self.syntax_reminder_prompt_template = get_prompts("syntax_reminder_prompt")
        self.operator_prompt_template = get_prompts("operator_prompt")

        self.code_reminder_prompt_template = get_prompts("code_reminder_prompt")
        self.programmer_prompt_template = get_prompts("programmer_prompt")
        self.evaluator_prompt_template = get_prompts("evaluator_prompt")

        self.test_prompt_template = get_prompts("test_prompt")

    def _init_fake_team_conversation(self, team_conversation, code_wo_labels):
        self.fake_team_conversation = copy.deepcopy(team_conversation)
        self.source_code = self.code_reminder_prompt_template.format(source_code=code_wo_labels)
        self.fake_team_conversation.append({"agent_name": 'Code reminder', "agent_response": self.source_code})

    def _init_cnt(self):
        self.syntax_cnt = 3
        self.operator_cnt = 3
        self.programmer_cnt = 3  # cnt for programmer output format
        self.evaluator_cnt = 3  # cnt for evaluator output format
        self.debug_times_left = 3  # cnt for debugging (format correct but not satisfactory code)
        self.syntax_success = False
        self.operator_success = False
        self.programmer_success = False
        self.evaluator_success = False

        self.queried_components = None
        self.queried_model = None
        self.queried_function = None

    def execute_code(self, revision_code, print_code):
        src_code = insert_code(self.source_code, revision_code, 'REVISION')
        src_code = insert_code(src_code, print_code, 'PRINT')
        execution_rst = run_with_exec(src_code)

        # save the complete code as .py
        with open(f"./logs/code_draft/complete_code_{self.debug_times_left}.py", "w") as f:
            f.write(src_code)
        # save the execution result as .txt
        with open(f"./logs/code_draft/execution_result_{self.debug_times_left}.txt", "w") as f:
            f.write(execution_rst)
        self.fake_team_conversation.append({"agent_name": 'Execution result', "agent_response": execution_rst})
        return src_code, execution_rst

    def tool_call_exp(self, prompt: Optional[str] = None, messages: Optional[List] = None,
                      seed: int = 10, temperature: float = 0.1,
                      is_syntax_guidance: bool = False,
                      syntax_mode: str = 'none'):

        # make sure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)
        # make sure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if not prompt is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        if is_syntax_guidance:
            tools = self.syntax_guidance_tool
            tool_choice = {"type": "function", "function": {"name": "syntax_guidance"}}
        else:
            if syntax_mode == 'multiple':
                tools = self.multiple_tools
            elif syntax_mode == 'single':
                tools = self.single_tools
            elif syntax_mode == 'none':
                tools = self.none_tools
            elif syntax_mode == 'all':
                tools = self.all_tools
            else:
                raise Exception("Invalid mode!")
            tool_choice = "required"

        if type(self.client) in [OpenAI, Client]:
            completion = self.client.chat.completions.create(
                model=self.llm,
                messages=messages,
                seed=seed,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice
            )
            if completion.choices[0].message.tool_calls:
                # internal tool is called
                fn_call = completion.choices[0].message.tool_calls[0].function
                fn_name = fn_call.name
                fn_args = fn_call.arguments
                print(f'function name = {fn_name}')
                print(f'function arguments = {fn_args}')
            else:
                raise Exception("No tool call executed by Operator, perhaps because of the 'auto' tool choice!")
        else:
            raise Exception("Client type not supported!")
        return fn_name, fn_args

    def generate_syntax_exp(self, args, messages, team_conversation, models_dict):
        while not self.syntax_success and self.syntax_cnt > 0:
            component_descriptions = extract_component_descriptions(models_dict)

            if models_dict['model_representation']['model type'] != 'LP':
                function_names = [fn for fn in self.function_names if fn != 'sensitivity_analysis']
            else:
                function_names = [fn for fn in self.function_names]

            prompt = self.syntax_reminder_prompt_template.format(function_names=function_names,
                                                                 component_name_meaning_pairs=str(component_descriptions))
            pseudo_messages = self.generate_pseudo_messages(messages, team_conversation, prompt)
            self.syntax_cnt -= 1
            try:
                syntax_start = time.time()
                fn_name, fn_args = self.tool_call_exp(messages=pseudo_messages,
                                                      seed=self.syntax_cnt, temperature=args.temperature,
                                                      is_syntax_guidance=True)
                syntax_end = time.time()
                self.syntax_time += (syntax_end - syntax_start)

                self.queried_function = json.loads(fn_args).get("queried_function")
                self.queried_components = json.loads(fn_args).get("queried_components")
                self.queried_model = json.loads(fn_args).get("queried_model")
                # forced syntax_guidance to be called
                syntax_output, syntax_mode = syntax_guidance(self.queried_function,
                                                             self.queried_components,
                                                             self.queried_model,
                                                             models_dict)
                self.syntax_success = True
                return syntax_output, syntax_mode

            except Exception as e:
                print(e)
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.syntax_cnt == 0:
                    self.syntax_success = False
                    return "LLM failed", "none"

    def generate_feedback_exp(self, args, messages, team_conversation, models_dict, syntax_mode):
        while not self.operator_success and self.operator_cnt > 0:
            prompt = self.operator_prompt_template  # nothing to format here
            pseudo_messages = self.generate_pseudo_messages(messages, team_conversation, prompt)
            self.operator_cnt -= 1
            try:
                syntax_start = time.time()
                fn_name, fn_args = self.tool_call_exp(messages=pseudo_messages,
                                                      seed=self.operator_cnt, temperature=args.temperature,
                                                      is_syntax_guidance=False,
                                                      syntax_mode=syntax_mode)
                syntax_end = time.time()
                self.syntax_time += (syntax_end - syntax_start)

                self.queried_function = fn_name
                self.queried_model = json.loads(fn_args).get("queried_model")
                self.unparsed_queried_components = json.loads(fn_args).get("queried_components")
                self.queried_components = fnArgsDecoder(self.unparsed_queried_components)

                # pass the function name and arguments to the function
                if fn_name == 'feasibility_restoration':
                    fn_output = feasibility_restoration(self.queried_components, self.queried_model, models_dict)
                elif fn_name == 'sensitivity_analysis':
                    fn_output = sensitivity_analysis(self.queried_components, self.queried_model, models_dict)
                elif fn_name == 'components_retrival':
                    fn_output = components_retrival(self.queried_components, self.queried_model, models_dict)
                elif fn_name == 'evaluate_modification':
                    fn_output = evaluate_modification(self.queried_components, self.queried_model, models_dict)
                else:
                    raise Exception("invalid function name")
                self.operator_success = True
                return fn_output

            except Exception as e:
                print(e)
                import traceback
                err = traceback.format_exc()
                # embed the error message into the syntax reminder in team_conversation
                error_response = f"\n\nProblematic queried_components: {self.unparsed_queried_components} \n\nError: {err}"
                team_conversation.append({"agent_name": 'Execution result', "agent_response": error_response})

                if self.operator_cnt == 0:
                    self.operator_success = False
                    return "LLM failed"

    def programmer_loop_exp(self, args, pseudo_messages):
        while self.programmer_cnt > 0:
            program_start = time.time()
            code_output = self.llm_call_exp(messages=pseudo_messages,
                                            seed=self.programmer_cnt, temperature=args.temperature, stream=False)
            program_end = time.time()
            self.programing_time += (program_end - program_start)

            self.programmer_cnt -= 1
            try:
                snippets = re.findall(self.pattern, code_output, flags=re.DOTALL)
                assert len(snippets) <= 2
                assert snippets[0][0] == 'python'
                assert snippets[1][0] == 'python'
                revision_code = snippets[0][1]
                print_code = snippets[1][1]
                self.programmer_success = True
                self.fake_team_conversation.append({"agent_name": 'Programmer', "agent_response": code_output})
                return code_output, revision_code, print_code

            except AssertionError as e:
                print(e)
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.programmer_cnt == 0:
                    self.programmer_success = False
                    return None, None, None

    def evaluator_loop_exp(self, args, pseudo_messages):
        while self.evaluator_cnt > 0:
            evaluation_start = time.time()
            evaluation_output = self.llm_call_exp(messages=pseudo_messages,
                                                  seed=self.evaluator_cnt, temperature=args.temperature,
                                                  json_mode=args.json_mode, stream=False)
            evaluation_end = time.time()
            self.evaluation_time += (evaluation_end - evaluation_start)

            self.evaluator_cnt -= 1
            try:
                evaluation = evaluation_output.strip()
                if "```json" in evaluation:
                    evaluation = evaluation.split("```json")[1].split("```")[0]
                evaluation = evaluation.replace("\\", "")
                # print('Code review:', evaluation)
                evaluation = json.loads(evaluation)
                decision = evaluation["decision"]
                comment = evaluation["comment"]
                self.evaluator_success = True
                self.fake_team_conversation.append({"agent_name": 'Evaluator', "agent_response": evaluation_output})
                return evaluation_output, decision, comment

            except AssertionError as e:
                print(e)
                # import traceback
                # err = traceback.format_exc()
                # print(err)
                if self.evaluator_cnt == 0:
                    self.evaluator_success = False
                    return None, None, None

    def generate_code_exp(self, args, messages, team_conversation, models_dict):
        # initialize
        self._init_prompt_template()
        self._init_fake_team_conversation(team_conversation, models_dict['model_representation']['code'])

        # until the programmer generates the code that evaluator approves
        while self.debug_times_left > 0:
            # Only init the cnt for programmer and evaluator for every debugging loop
            # because _init_cnt() will reset all the success, cnt, debug_times_left
            self.programmer_cnt = 3
            self.evaluator_cnt = 3
            # until the programmer generates the code in correct format
            programmer_prompt = self.programmer_prompt_template
            pseudo_messages = self.generate_pseudo_messages(messages, self.fake_team_conversation, programmer_prompt)
            code_output, revision_code, print_code = self.programmer_loop_exp(args, pseudo_messages)
            if not self.programmer_success:
                return "LLM failed", 'None', 'None'

            # simply executing the code
            complete_code, execution_rst = self.execute_code(revision_code, print_code)

            # until the evaluator evaluates the code in correct format
            evaluator_prompt = self.evaluator_prompt_template
            pseudo_messages = self.generate_pseudo_messages(messages, self.fake_team_conversation, evaluator_prompt)
            evaluation_output, decision, comment = self.evaluator_loop_exp(args, pseudo_messages)
            if not self.evaluator_success:
                return code_output, execution_rst, "LLM failed"

            if decision == 'accept':
                return code_output, execution_rst, evaluation_output
            else:
                self.debug_times_left -= 1
                if self.debug_times_left == 0:
                    # return the last evaluation output though it is rejected by evaluator
                    return code_output, execution_rst, evaluation_output

    def generate_report_exp(self, args, messages, team_conversation, models_dict):
        self._init_cnt()

        if args.external_experiment:
            syntax_output, syntax_mode = 'external_tools', 'none'
            self.syntax_success = True
        else:
            syntax_output, syntax_mode = self.generate_syntax_exp(args, messages, team_conversation, models_dict)

        if not self.syntax_success:
            team_conversation.append({"agent_name": 'Syntax reminder', "agent_response": syntax_output})
            messages.append({"role": "assistant", "content": syntax_output})
        else:
            if syntax_output != 'external_tools':
                # add code reminder to the team_conversation as well to help find correct component indexes
                # add syntax reminder
                team_conversation.append({"agent_name": 'Code reminder',
                                          "agent_response": models_dict['model_representation']['code']})
                team_conversation.append({"agent_name": 'Syntax reminder', "agent_response": syntax_output})
                function_output = self.generate_feedback_exp(args, messages, team_conversation, models_dict,
                                                             syntax_mode)

                team_conversation = [item for item in team_conversation if
                                     item["agent_name"] not in ['Code reminder', 'Syntax reminder']]

                team_conversation.append({"agent_name": 'Operator', "agent_response": function_output})
                if self.operator_success:
                    messages.append({"role": "assistant", "content": function_output})
                else:
                    syntax_output = 'external_tools'

            if not args.internal_experiment:
                if syntax_output == 'external_tools':
                    code_output, execution_rst, evaluation_output = self.generate_code_exp(args, messages,
                                                                                           team_conversation,
                                                                                           models_dict)
                    team_conversation.append({"agent_name": 'Programmer', "agent_response": code_output})
                    team_conversation.append({"agent_name": 'Execution result', "agent_response": execution_rst})
                    team_conversation.append({"agent_name": 'Evaluator', "agent_response": evaluation_output})

                    messages.append({"role": "assistant", "content": "Programmer:\n\n" + code_output})
                    messages.append({"role": "assistant", "content": "Execution result:\n\n" + execution_rst})
                    messages.append({"role": "assistant", "content": "Evaluator:\n\n" + evaluation_output})

        return messages, team_conversation

    def generate_test_result_exp(self, args, messages, gt_a):
        self._init_prompt_template()
        prompt = self.test_prompt_template.format(human_expert_answer=gt_a)
        pseudo_messages = self.generate_pseudo_messages(messages, [], prompt)
        pass_or_fail = self.llm_call_exp(messages=pseudo_messages, temperature=args.temperature, stream=False)
        return pass_or_fail