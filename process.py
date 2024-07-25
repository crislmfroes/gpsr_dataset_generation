from distilabel.llms import TransformersLLM, OllamaLLM, MistralLLM
from distilabel.steps.tasks.structured_generation import StructuredGeneration
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict
from distilabel.steps.generators.huggingface import LoadDataFromHub
from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps import Step
from distilabel.steps import StepInput, StepOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline import Pipeline
from typing import List, Dict, Union, Literal, Type, Any
import config
from datasets import load_dataset
from pydantic import BaseModel, Field


class RobotState(BaseModel):
    state: str = Field(description=f"Must be one of: {['speak', 'move_to', 'grasp', 'pass_to', 'follow', 'answer', 'visual_question_answering']}")
    state_arg: str

class SequentialStateMachine(BaseModel):
    plan: str
    states: List[RobotState]

examples = [
    {
        'user': 'move the apple from the desk to the table',
        'assistant': {
            'plan': 'I must go to the apple on the desk, grasp it, then go to the table and pass the apple to the table',
            'states': [
                {'state': 'move_to', 'state_arg': 'apple on the desk'},
                {'state': 'grasp', 'state_arg': 'apple'},
                {'state': 'move_to', 'state_arg': 'table'},
                {'state': 'pass_to', 'state_arg': 'table'}
            ]
        }
    }
]

'''class SetSystemPromptStep(Step):
    examples: List[Dict[str, Any]]

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return ["system_prompt"]

    def process(self, inputs: StepInput) -> StepOutput:
        #schema.pop('$defs')
        for i, input in enumerate(inputs):
            #input["instruction"] = {'role': 'user', 'content': input['instruction']}
            #input['system_prompt'] = {'role': 'system', 'content': 'Answer by providing a schema for a state machine in JSON format.'}
            input['system_prompt'] = f"You must always answer the user by thinking step-by-step and generating a state machine that will allow a robot to accomplish he user given task. The following skills are available to the robot: {', '.join(['speak', 'move_to', 'grasp', 'pass_to', 'follow', 'answer', 'visual_question_answering'])}. Examples: {self.examples}"
            inputs[i] = input
        yield inputs'''
#schema = jsonref.loads(SequentialStateMachine.schema_json())
#schema.pop('$defs')
with Pipeline(name='gpsr-synthetic-data', description='generate synthetic data for GPSR') as pipe:
    dataset = load_dataset(path=config.command_dataset, split='train')
    load_dataset = LoadDataFromDicts(
        name='load-dataset',
        data=[{
            "instruction": row['instruction'],
            "system_prompt": f"Answer by generating a state machine that will allow a robot to perform the user given task. Examples: {examples}",
            "structured_output": {
                "format": "json",
                "schema": SequentialStateMachine.model_json_schema()
            }
        } for row in dataset],
        #batch_size=1,
    )
    if config.llm_api_type == 'mistral':
        llm = MistralLLM(
            model=config.model,
            max_concurrent_requests=1
        )
    elif config.llm_api_type == 'ollama':
        llm = OllamaLLM(
            model=config.model
        )
    generate = StructuredGeneration(
        name='generate',
        use_system_prompt=True,
        llm=llm,
    )
    #load_dataset >> set_system >> generate
    load_dataset >> generate

if __name__ == '__main__':
    #print(StateMachine.model_json_schema().keys())
    #exit()
    distiset = pipe.run()
    #print(distiset['default']['train'][0]['generation'])
    #exit()
    distiset.push_to_hub(config.output_dataset)