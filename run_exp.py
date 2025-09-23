import os
import time
import datetime
import json
import pickle
import csv
from extractor import (get_files, get_files_generator, initial_loading, update_model_representation,
                       get_skipJSON, feed_skipJSON)
from internal_tools import feasibility_restoration, sensitivity_analysis, components_retrival, evaluate_modification
from utils import get_agents, OptiChat_workflow_exp
from pyomo.opt import TerminationCondition
from openai import OpenAI


class Args:
    def __init__(self,
                 exps_id,
                 folder_name,
                 gpt_model,
                 temperature,
                 json_mode,
                 skip_syntax,
                 skip_description,
                 interpreter_experiment,
                 internal_experiment,
                 external_experiment,
                 ablation
                 ):
        self.exps_id = exps_id
        self.folder_name = folder_name
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.json_mode = json_mode
        self.skip_syntax = skip_syntax
        self.skip_description = skip_description
        self.interpreter_experiment = interpreter_experiment
        self.internal_experiment = internal_experiment
        self.external_experiment = external_experiment
        self.ablation = ablation
        self.illustration_stream = False
        self.inference_stream = False
        self.explanation_stream = False
        self.fn_names = ["feasibility_restoration", "sensitivity_analysis", "components_retrival", "evaluate_modification", "external_tools"]

    def __str__(self):
        return ', '.join(f'{k}={v}' for k, v in self.__dict__.items())


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def run_interpretation_experiment(args):
    """
    run experiments on interpretation,
    save the model_json so that internal and external experiments can load the descriptions directly
    """
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}
    for file in files_generator:
        file_stats = {}
        print('='*10)
        print(f'interpretation experiment: {file}')

        try:
            # component interpretation
            models_dict, code = initial_loading(file, is_uploaded=False)
            start = time.time()
            models_dict, cnt, completion = interpreter.generate_interpretation_exp(args, models_dict, code)
            end = time.time()
            file_stats['component interpretation time'] = end - start
            file_stats['cnt'] = (cnt + 1) / 3
            file_stats['success'] = completion

            # update model representation with component descriptions
            update_model_representation(models_dict)

            # model interpretation
            start = time.time()
            illustration = interpreter.generate_illustration_exp(args, models_dict["model_representation"])
            end = time.time()
            file_stats['model interpretation time'] = end - start
            file_stats['total time'] = (file_stats['component interpretation time'] +
                                        file_stats['model interpretation time'])

            # update model representation with model description
            models_dict['model_1']['model description'] = illustration
            update_model_representation(models_dict)

            # infeasibility interpretation (if true)
            if models_dict['model_1']['model status'] in [TerminationCondition.infeasible,
                                                          TerminationCondition.infeasibleOrUnbounded]:
                start = time.time()
                inference = interpreter.generate_inference_exp(args, models_dict["model_representation"])
                end = time.time()
                file_stats['model inference time'] = end - start
                file_stats['total time'] += file_stats['model inference time']

                # update model representation with inference description (in model description)
                models_dict['model_1']['model description'] = illustration + '\n' + inference
                update_model_representation(models_dict)

            # save
            skipJSON = get_skipJSON(models_dict["model_representation"])
            with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "w") as f:
                json.dump(skipJSON, f)
            # update stats
            stats[file] = file_stats

        except Exception as e:
            print(f'Error during interpretation of {file}:', e)
            # update files with error
            files_with_error[file] = str(e)

        # save stats
        with open(f'logs/stats/{args.folder_name}/interpretation/{args.exps_id}/stats.json', 'w') as f:
            json.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/interpretation/{args.exps_id}/files_with_error.json', 'w') as f:
            json.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", len(files_with_error))


def run_internal_experiment(args):
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'internal experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_queried_components = qa["queried_components"]
                    q = qa["Q"]

                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    # workflow
                    updated_messages, team_conversation = OptiChat_workflow_exp(args, coordinator, engineer, explainer,
                                                                                messages, models_dict)

                    # record the results of each question
                    qa_stats["queried_components"] = gt_queried_components
                    qa_stats["Q"] = q

                    qa_stats["llm_queried_components"] = engineer.queried_components
                    qa_stats["llm_queried_function"] = engineer.queried_function
                    # record the measures of each question
                    qa_stats["syntax time"] = engineer.syntax_time
                    qa_stats["coordination time"] = coordinator.coordination_time
                    qa_stats["explanation time"] = explainer.explanation_time
                    qa_stats["total time"] = (engineer.syntax_time +
                                        coordinator.coordination_time +
                                        explainer.explanation_time)
                    # note that syntax cnt is usually 3/3 because identifying name, model, anf function name is easy
                    qa_stats["syntax cnt"] = (engineer.syntax_cnt + 1) / 3
                    # operator cnt can be very small, such as 2/3 and 1/3, because it is hard to identify the indexes
                    # improve prompt and workflow to make this measure higher
                    qa_stats["operator cnt"] = (engineer.operator_cnt + 1) / 3
                    # coordinator cnt is usually 3/3 because the coordinator can always generate a decision easily
                    # note that coordinator made 2 decisions, first engineer then explainer, here takes the second decision
                    # actually coordinator.cnt doesn't need to add 1 because of the cnt-1 in the workflow
                    qa_stats["coordinator cnt"] = (coordinator.coordinator_cnt + 1) / 3
                    qa_stats["coordinator success"] = coordinator.coordinator_success
                    # note that syntax success means syntax reminder and operator are both successful,
                    # in contrast, syntax cnt means cnt in syntax reminder only, operator cnt means cnt in operator only
                    qa_stats["syntax success"] = (engineer.syntax_success and engineer.operator_success)
                    qa_stats["args correct"] = (engineer.queried_components == gt_queried_components)
                    qa_stats["fn correct"] = (engineer.queried_function == task)
                    qa_stats["success"] = (qa_stats["args correct"] and qa_stats["fn correct"])
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation
                except Exception as e:
                    print(f'Error during internal tool experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/internal/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/internal/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))


def run_internal_ablation_experiment(args):
    args.internal_experiment = False
    args.external_experiment = True  # to start code generation

    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'internal experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_queried_components = qa["queried_components"]
                    q = qa["Q"]

                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    team_conversation = []
                    engineer.syntax_time = 0
                    engineer.programing_time = 0
                    engineer.evaluation_time = 0

                    updated_messages, team_conversation = engineer.generate_report_exp(args, messages,
                                                                                       team_conversation, models_dict)

                    queried_function = task
                    queried_model = "model_1"
                    queried_components = gt_queried_components
                    # pass the function name and arguments to the function
                    if queried_function == 'feasibility_restoration':
                        fn_output = feasibility_restoration(queried_components, queried_model, models_dict)
                    elif queried_function == 'sensitivity_analysis':
                        fn_output = sensitivity_analysis(queried_components, queried_model, models_dict)
                    elif queried_function == 'components_retrival':
                        fn_output = components_retrival(queried_components, queried_model, models_dict)
                    elif queried_function == 'evaluate_modification':
                        fn_output = evaluate_modification(queried_components, queried_model, models_dict)
                    else:
                        raise Exception("invalid function name")
                    pass_or_fail = engineer.generate_test_result_exp(args, updated_messages, fn_output)

                    # record the results of each question
                    qa_stats["queried_components"] = gt_queried_components
                    qa_stats["Q"] = q
                    qa_stats["gt_a"] = fn_output
                    qa_stats["llm_code"] = team_conversation[-3]["agent_response"]  # Programmer
                    qa_stats["llm_a"] = team_conversation[-2]["agent_response"]  # Execution result
                    qa_stats["llm_evaluation"] = team_conversation[-1]["agent_response"]  # Evaluator
                    qa_stats["llm_pass_or_fail"] = pass_or_fail
                    # record the measures of each question
                    qa_stats["programing time"] = engineer.programing_time
                    qa_stats["evaluation time"] = engineer.evaluation_time
                    qa_stats["total time"] = engineer.programing_time + engineer.evaluation_time
                    qa_stats["debug cnt"] = (engineer.debug_times_left + 1) / 3
                    qa_stats["success"] = True if qa_stats["llm_pass_or_fail"].lower() == "pass" else False  # auto
                    qa_stats["manual success"] = None  # manual inspection on llm_a and llm_evaluation
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation

                except Exception as e:
                    print(f'Error during internal tool experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/internal_ablation/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/internal_ablation/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))


def run_internal_wo_syntax_experiment(args):
    assert args.skip_syntax is True
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'internal experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_queried_components = qa["queried_components"]
                    q = qa["Q"]

                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    # workflow
                    updated_messages, team_conversation = OptiChat_workflow_exp(args, coordinator, engineer, explainer,
                                                                                messages, models_dict)

                    # record the results of each question
                    qa_stats["queried_components"] = gt_queried_components
                    qa_stats["Q"] = q

                    qa_stats["llm_queried_components"] = engineer.queried_components
                    qa_stats["llm_queried_function"] = engineer.queried_function
                    # record the measures of each question
                    qa_stats["syntax time"] = engineer.syntax_time
                    qa_stats["coordination time"] = coordinator.coordination_time
                    qa_stats["explanation time"] = explainer.explanation_time
                    qa_stats["total time"] = (engineer.syntax_time +
                                        coordinator.coordination_time +
                                        explainer.explanation_time)
                    # note that syntax cnt is usually 3/3 because identifying name, model, anf function name is easy
                    qa_stats["syntax cnt"] = (engineer.syntax_cnt + 1) / 3
                    # operator cnt can be very small, such as 2/3 and 1/3, because it is hard to identify the indexes
                    # improve prompt and workflow to make this measure higher
                    qa_stats["operator cnt"] = (engineer.operator_cnt + 1) / 3
                    # coordinator cnt is usually 3/3 because the coordinator can always generate a decision easily
                    # note that coordinator made 2 decisions, first engineer then explainer, here takes the second decision
                    # actually coordinator.cnt doesn't need to add 1 because of the cnt-1 in the workflow
                    qa_stats["coordinator cnt"] = (coordinator.coordinator_cnt + 1) / 3
                    qa_stats["coordinator success"] = coordinator.coordinator_success
                    # note that syntax success means syntax reminder and operator are both successful,
                    # in contrast, syntax cnt means cnt in syntax reminder only, operator cnt means cnt in operator only
                    qa_stats["syntax success"] = (engineer.syntax_success and engineer.operator_success)
                    qa_stats["args correct"] = (engineer.queried_components == gt_queried_components)
                    qa_stats["fn correct"] = (engineer.queried_function == task)
                    qa_stats["success"] = (qa_stats["args correct"] and qa_stats["fn correct"])
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation
                except Exception as e:
                    print(f'Error during internal tool experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/internal_wo/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/internal_wo/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))


def run_internal_wo_description_experiment(args):
    def remove_description(models_dict):
        """
        Recursively remove description values from models_dict by setting them to empty strings.
        This function modifies the dictionary in place and also returns it.
        Args:
            models_dict: Dictionary that may contain nested dictionaries with description keys
        Returns:
            The modified dictionary with description values set to empty strings
        """
        if isinstance(models_dict, dict):
            for key, value in models_dict.items():
                if "description" in key.lower():
                    models_dict[key] = ""
                else:
                    remove_description(value)
        elif isinstance(models_dict, (list, tuple)):
            for item in models_dict:
                remove_description(item)
        return models_dict

    assert args.skip_description is True
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'internal experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        remove_description(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_queried_components = qa["queried_components"]
                    q = qa["Q"]

                    print(f"--- removed description from models_dict for file {file}: "
                          f"{models_dict['model_representation']['model description']}")
                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    # workflow
                    updated_messages, team_conversation = OptiChat_workflow_exp(args, coordinator, engineer, explainer,
                                                                                messages, models_dict)

                    # record the results of each question
                    qa_stats["queried_components"] = gt_queried_components
                    qa_stats["Q"] = q

                    qa_stats["llm_queried_components"] = engineer.queried_components
                    qa_stats["llm_queried_function"] = engineer.queried_function
                    # record the measures of each question
                    qa_stats["syntax time"] = engineer.syntax_time
                    qa_stats["coordination time"] = coordinator.coordination_time
                    qa_stats["explanation time"] = explainer.explanation_time
                    qa_stats["total time"] = (engineer.syntax_time +
                                        coordinator.coordination_time +
                                        explainer.explanation_time)
                    # note that syntax cnt is usually 3/3 because identifying name, model, anf function name is easy
                    qa_stats["syntax cnt"] = (engineer.syntax_cnt + 1) / 3
                    # operator cnt can be very small, such as 2/3 and 1/3, because it is hard to identify the indexes
                    # improve prompt and workflow to make this measure higher
                    qa_stats["operator cnt"] = (engineer.operator_cnt + 1) / 3
                    # coordinator cnt is usually 3/3 because the coordinator can always generate a decision easily
                    # note that coordinator made 2 decisions, first engineer then explainer, here takes the second decision
                    # actually coordinator.cnt doesn't need to add 1 because of the cnt-1 in the workflow
                    qa_stats["coordinator cnt"] = (coordinator.coordinator_cnt + 1) / 3
                    qa_stats["coordinator success"] = coordinator.coordinator_success
                    # note that syntax success means syntax reminder and operator are both successful,
                    # in contrast, syntax cnt means cnt in syntax reminder only, operator cnt means cnt in operator only
                    qa_stats["syntax success"] = (engineer.syntax_success and engineer.operator_success)
                    qa_stats["args correct"] = (engineer.queried_components == gt_queried_components)
                    qa_stats["fn correct"] = (engineer.queried_function == task)
                    qa_stats["success"] = (qa_stats["args correct"] and qa_stats["fn correct"])
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation
                except Exception as e:
                    print(f'Error during internal tool experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/internal_wo2/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/internal_wo2/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))


def validate_internal_testset(args):
    args.internal_experiment = False
    args.external_experiment = True  # to start code generation

    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)

    errors = []

    for file in files_generator:
        print('='*10)
        print(f'internal experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/tool_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            for i, qa in enumerate(qas):
                try:
                    # get ground truth
                    gt_queried_components = qa["queried_components"]

                    queried_function = task
                    queried_model = "model_1"
                    queried_components = gt_queried_components
                    # pass the function name and arguments to the function
                    if queried_function == 'feasibility_restoration':
                        fn_output = feasibility_restoration(queried_components, queried_model, models_dict)
                    elif queried_function == 'sensitivity_analysis':
                        fn_output = sensitivity_analysis(queried_components, queried_model, models_dict)
                    elif queried_function == 'components_retrival':
                        fn_output = components_retrival(queried_components, queried_model, models_dict)
                    elif queried_function == 'evaluate_modification':
                        fn_output = evaluate_modification(queried_components, queried_model, models_dict)
                    else:
                        raise Exception("invalid function name")

                except Exception as e:
                    print(f'Error during internal tool experiments of {file}:', e)
                    errors.append({"file": file, "task": task, "qa": qa, "error": str(e)})
    print(f"Total errors encountered: {len(errors)}")
    for error in errors:
        print("===="*10)
        print(error)




def run_external_experiment(args):
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'external experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/code_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/code_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_a = qa["A"]
                    q = qa["Q"]

                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    team_conversation = []
                    engineer.syntax_time = 0
                    engineer.programing_time = 0
                    engineer.evaluation_time = 0

                    updated_messages, team_conversation = engineer.generate_report_exp(args, messages,
                                                                                       team_conversation, models_dict)

                    pass_or_fail = engineer.generate_test_result_exp(args, updated_messages, gt_a)

                    # record the results of each question
                    qa_stats["Q"] = q
                    qa_stats["gt_a"] = gt_a
                    qa_stats["llm_code"] = team_conversation[-3]["agent_response"]  # Programmer
                    qa_stats["llm_a"] = team_conversation[-2]["agent_response"]  # Execution result
                    qa_stats["llm_evaluation"] = team_conversation[-1]["agent_response"]  # Evaluator
                    qa_stats["llm_pass_or_fail"] = pass_or_fail
                    # record the measures of each question
                    qa_stats["programing time"] = engineer.programing_time
                    qa_stats["evaluation time"] = engineer.evaluation_time
                    qa_stats["total time"] = engineer.programing_time + engineer.evaluation_time
                    qa_stats["debug cnt"] = (engineer.debug_times_left + 1) / 3
                    qa_stats["success"] = True if qa_stats["llm_pass_or_fail"].lower() == "pass" else False  # auto
                    qa_stats["manual success"] = None  # manual inspection on llm_a and llm_evaluation
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation

                except Exception as e:
                    print(f'Error during external code experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/external/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/external/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))


def run_external_wo_description_experiment(args):
    def remove_description(models_dict):
        """
        Recursively remove description values from models_dict by setting them to empty strings.
        This function modifies the dictionary in place and also returns it.
        Args:
            models_dict: Dictionary that may contain nested dictionaries with description keys
        Returns:
            The modified dictionary with description values set to empty strings
        """
        if isinstance(models_dict, dict):
            for key, value in models_dict.items():
                if "description" in key.lower():
                    models_dict[key] = ""
                else:
                    remove_description(value)
        elif isinstance(models_dict, (list, tuple)):
            for item in models_dict:
                remove_description(item)
        return models_dict

    assert args.skip_description is True
    stats = {}
    interpreter, explainer, engineer, coordinator = get_agents(args.fn_names, client, args.gpt_model)
    files_generator = get_files_generator(args.folder_name)
    files_with_error = {}

    for file in files_generator:
        file_stats = {}
        files_with_error[file] = []
        print('='*10)
        print(f'external experiment: {file}')

        # load model
        models_dict, code = initial_loading(file, is_uploaded=False)
        if not os.path.exists(f"logs/model_json/{os.path.splitext(file)[0]}.json"):
            raise FileNotFoundError(f"Model JSON file not found for {file}. Run interpretation experiment first.")
        with open(f"logs/model_json/{os.path.splitext(file)[0]}.json", "r") as f:
            skipJSON = json.load(f)
        models_dict = feed_skipJSON(skipJSON, models_dict)
        update_model_representation(models_dict)
        remove_description(models_dict)
        # load corresponding QA
        if not os.path.exists(f'test_set/code_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl'):
            print(f"Test set file not found for {file}. Skip")
            continue
        with open(f'test_set/code_testset/{os.path.basename(os.path.splitext(file)[0])}.pkl', 'rb') as f:
            test_set = pickle.load(f)
        for task, qas in test_set.items():
            file_stats[task] = []
            for qa in qas:
                qa_stats = {}
                try:
                    # get ground truth
                    gt_a = qa["A"]
                    q = qa["Q"]

                    print(f"--- removed description from models_dict for file {file}: ")
                    # start with an empty file-specific message, load the interpretation for each question
                    messages = [{"role": "user", "content": "I have uploaded a Pyomo model."},
                                {"role": "assistant",
                                 "content": models_dict["model_representation"]["model description"]},
                                {"role": "user", "content": q}]

                    team_conversation = []
                    engineer.syntax_time = 0
                    engineer.programing_time = 0
                    engineer.evaluation_time = 0

                    updated_messages, team_conversation = engineer.generate_report_exp(args, messages,
                                                                                       team_conversation, models_dict)

                    pass_or_fail = engineer.generate_test_result_exp(args, updated_messages, gt_a)

                    # record the results of each question
                    qa_stats["Q"] = q
                    qa_stats["gt_a"] = gt_a
                    qa_stats["llm_code"] = team_conversation[-3]["agent_response"]  # Programmer
                    qa_stats["llm_a"] = team_conversation[-2]["agent_response"]  # Execution result
                    qa_stats["llm_evaluation"] = team_conversation[-1]["agent_response"]  # Evaluator
                    qa_stats["llm_pass_or_fail"] = pass_or_fail
                    # record the measures of each question
                    qa_stats["programing time"] = engineer.programing_time
                    qa_stats["evaluation time"] = engineer.evaluation_time
                    qa_stats["total time"] = engineer.programing_time + engineer.evaluation_time
                    qa_stats["debug cnt"] = (engineer.debug_times_left + 1) / 3
                    qa_stats["success"] = True if qa_stats["llm_pass_or_fail"].lower() == "pass" else False  # auto
                    qa_stats["manual success"] = None  # manual inspection on llm_a and llm_evaluation
                    # save the team conversation for manual inspection
                    qa_stats['team_conversation'] = team_conversation

                except Exception as e:
                    print(f'Error during external code experiments of {file}:', e)
                    # update files with error
                    files_with_error[file].append({'error': str(e), 'task': task, 'qa': qa})
                file_stats[task].append(qa_stats)
        # update stats
        stats[file] = file_stats
        # save stats
        with open(f'logs/stats/{args.folder_name}/external_wo2/{args.exps_id}/stats_detail.pkl', 'wb') as f:
            pickle.dump(stats, f)
        # save files_with_error
        with open(f'logs/stats/{args.folder_name}/external_wo2/{args.exps_id}/files_with_error.pkl', 'wb') as f:
            pickle.dump(files_with_error, f)
        print(f'File {file} processed. Stats: {file_stats}')
    print("number of files without error:", len(stats))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))



def create_internal_report(args):
    args_dict = vars(args)

    if args.skip_syntax:
        postfix = "_wo"
    elif args.skip_description:
        postfix = "_wo2"
    else:
        postfix = ""

    with open(f'logs/stats/{args.folder_name}/internal{postfix}/{args.exps_id}/stats_detail.pkl', 'rb') as f:
        stats_detail = pickle.load(f)
    with open(f'logs/stats/{args.folder_name}/internal{postfix}/{args.exps_id}/files_with_error.pkl', 'rb') as f:
        files_with_error = pickle.load(f)
    print("number of files without error:", len(stats_detail))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))
    if sum(len(v) for k, v in files_with_error.items()) > 0:
        print("Files with errors:")
        print(files_with_error)
        raise ValueError("There are files with errors, please check the logs.")
    stats = {task: {"success": 0, "fn correct": 0, "args correct": 0, "num q": 0, "total time": 0} for task in args.fn_names}
    for file, tasks in stats_detail.items():
        for task, file_stats in tasks.items():
            for qa_stats in file_stats:
                stats[task]["num q"] += 1
                stats[task]["fn correct"] += qa_stats['fn correct']
                stats[task]["args correct"] += qa_stats['args correct']
                stats[task]["success"] += qa_stats['success']
                stats[task]["total time"] += qa_stats['total time']
    flattened_stats = {}
    for task in stats:
        if stats[task]["num q"] > 0:
            flattened_stats[f"{task}_fn_correct"] = stats[task]["fn correct"] / stats[task]["num q"]
            flattened_stats[f"{task}_args_correct"] = stats[task]["args correct"] / stats[task]["num q"]
            flattened_stats[f"{task}_success"] = stats[task]["success"] / stats[task]["num q"]
            flattened_stats[f"{task}_total_time"] = stats[task]["total time"] / stats[task]["num q"]
    for k, v in flattened_stats.items():
        print(f"{k}: {v:.4f}")

    report = args_dict | flattened_stats
    with open(f'logs/stats/{args.folder_name}/internal{postfix}/{args.exps_id}/stats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value"])  # write the header
        for key, value in report.items():
            writer.writerow([key, value])


def create_external_report(args, exp_type):
    args_dict = vars(args)

    assert exp_type in ["external", "internal_ablation"]
    if args.skip_description:
        postfix = "_wo2"
    else:
        postfix = ""

    if not os.path.exists(f'logs/stats/{args.folder_name}/{exp_type}{postfix}/{args.exps_id}/graded_stats_detail.pkl'):
        raise FileNotFoundError(f"Graded stats detail file not found for {args.folder_name} and {args.exps_id}. Grade it first.")

    with open(f'logs/stats/{args.folder_name}/{exp_type}{postfix}/{args.exps_id}/graded_stats_detail.pkl', 'rb') as f:
        stats_detail = pickle.load(f)
    with open(f'logs/stats/{args.folder_name}/{exp_type}{postfix}/{args.exps_id}/files_with_error.pkl', 'rb') as f:
        files_with_error = pickle.load(f)
    print("number of files without error:", len(stats_detail))
    print("number of files with error:", sum(len(v) for k, v in files_with_error.items()))
    if sum(len(v) for k, v in files_with_error.items()) > 0:
        print("Files with errors:")
        print(files_with_error)
        raise ValueError("There are files with errors, please check the logs.")
    if exp_type == "external":
        stats = {task: {"unverified success": 0, "verified success": 0, "num q": 0, "total time": 0} for task in ["external_tools"]}
    else:
        stats = {task: {"unverified success": 0, "verified success": 0, "num q": 0, "total time": 0} for task in args.fn_names}
    for file, tasks in stats_detail.items():
        for task, file_stats in tasks.items():
            for qa_stats in file_stats:
                stats[task]["num q"] += 1
                stats[task]["unverified success"] += qa_stats['success']
                stats[task]["verified success"] += qa_stats['manual success']
                stats[task]["total time"] += qa_stats['total time']
    flattened_stats = {}
    for task in stats:
        if stats[task]["num q"] > 0:
            flattened_stats[f"{task}_unverified_success"] = stats[task]["unverified success"] / stats[task]["num q"]
            flattened_stats[f"{task}_verified_success"] = stats[task]["verified success"] / stats[task]["num q"]
            flattened_stats[f"{task}_total_time"] = stats[task]["total time"] / stats[task]["num q"]
    for k, v in flattened_stats.items():
        print(f"{k}: {v:.4f}")

    report = args_dict | flattened_stats
    with open(f'logs/stats/{args.folder_name}/{exp_type}{postfix}/{args.exps_id}/stats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value"])  # write the header
        for key, value in report.items():
            writer.writerow([key, value])



if __name__ == '__main__':
    create_report_only = True

    exps_id = datetime.datetime.now().strftime("%m%d%H%M%S")
    folder_name = "Feas"  # Infeas, Feas
    interpreter_experiment = False  # run the interpreter experiment
    internal_experiment = False  # run the internal experiment
    external_experiment = True  # run the external experiment
    ablation = False
    skip_syntax = False
    skip_description = True

    gpt_model = 'gpt-4.1'  # 'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', "gpt-4.1", "o3"
    temperature = 0  # temperature for the model

    if sum([interpreter_experiment, internal_experiment, external_experiment]) > 1:
        raise ValueError("Run only one experiment at a time.")
    if sum([ablation, skip_syntax, skip_description]) > 1:
        raise ValueError("Run only one ablation experiment at a time.")
    if interpreter_experiment:
        json_mode = True
    elif internal_experiment:
        if ablation:
            json_mode = True
        else:
            json_mode = True
    elif external_experiment:
        json_mode = True

    args = Args(exps_id=exps_id,
                folder_name=folder_name,
                gpt_model=gpt_model,
                temperature=temperature,
                json_mode=json_mode,
                skip_syntax=skip_syntax,
                skip_description=skip_description,
                interpreter_experiment=interpreter_experiment,
                internal_experiment=internal_experiment,
                external_experiment=external_experiment,
                ablation=ablation
                )
    print(f"{k}: {v}" for k, v in vars(args).items())
    print('='*20)

    if args.interpreter_experiment:
        os.makedirs(f'logs/model_json/{args.folder_name}', exist_ok=True)
        os.makedirs(f'logs/stats/{args.folder_name}/interpretation/{args.exps_id}', exist_ok=True)
        if not create_report_only:
            run_interpretation_experiment(args)
    if args.internal_experiment:
        #validate_internal_testset(args)
        if not create_report_only:
            if ablation:
                os.makedirs(f'logs/stats/{args.folder_name}/internal_ablation/{args.exps_id}', exist_ok=True)
                run_internal_ablation_experiment(args)
            elif skip_syntax:
                os.makedirs(f'logs/stats/{args.folder_name}/internal_wo/{args.exps_id}', exist_ok=True)
                run_internal_wo_syntax_experiment(args)
            elif skip_description:
                os.makedirs(f'logs/stats/{args.folder_name}/internal_wo2/{args.exps_id}', exist_ok=True)
                run_internal_wo_description_experiment(args)
            else:
                os.makedirs(f'logs/stats/{args.folder_name}/internal/{args.exps_id}', exist_ok=True)
                run_internal_experiment(args)
        if not ablation:
            create_internal_report(args)
        else:
            if create_report_only:
                create_external_report(args, "internal_ablation")
            else:
                raise ValueError("Need Manual Verification. Skipping internal report creation for ablation experiment.")

    if args.external_experiment:
        if not create_report_only:
            if skip_description:
                os.makedirs(f'logs/stats/{args.folder_name}/external_wo2/{args.exps_id}', exist_ok=True)
                run_external_wo_description_experiment(args)
            else:
                os.makedirs(f'logs/stats/{args.folder_name}/external/{args.exps_id}', exist_ok=True)
                run_external_experiment(args)
            print("Need Manual Verification. Skipping external report creation for ablation experiment.")
        create_external_report(args, "external")





