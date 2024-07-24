from pydantic import BaseModel, Field
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks.structured_generation import StructuredGeneration
from distilabel.steps.tasks.text_generation import TextGeneration
from distilabel.steps.tasks.structured_outputs.utils import schema_as_dict
from distilabel.steps.generators.huggingface import LoadDataFromHub
from distilabel.steps.decorator import step
from distilabel.steps import StepInput, StepOutput
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.pipeline import Pipeline
from typing import List, Dict, Union, Literal


class RobotState(BaseModel):
    state: Literal['speak', 'move_to', 'grasp', 'pass_to', 'follow', 'answer', 'visual_question_answering']
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

@step(inputs=["instruction",], outputs=["instruction", "structured_output", "system_prompt"])
def SetSchemaStep(inputs: StepInput) -> StepOutput:
    schema = StateMachine.model_json_schema()
    #schema.pop('$defs')
    for i, input in enumerate(inputs):
        input["structured_output"] = {
            'format': 'json',
            'schema': schema
        }
        #input["instruction"] = {'role': 'user', 'content': input['instruction']}
        #input['system_prompt'] = {'role': 'system', 'content': 'Answer by providing a schema for a state machine in JSON format.'}
        input['system_prompt'] = f"You must always answer the user by thinking step-by-step and generating a state machine that will allow a robot to accomplish he user given task. Examples: {examples}"
        input['instruction'] = f'{input["instruction"]}.'
        inputs[i] = input
    yield inputs

with Pipeline(name='gpsr-synthetic-data', description='generate synthetic data for GPSR') as pipe:
    
    load_dataset = LoadDataFromHub(
        name='load-dataset',
        repo_id="crislmfroes/egpsr_commands",
        split="train",
        #batch_size=2,
    )
    set_schema = SetSchemaStep(
        name='set-schema',
        #input_batch_size=2,
    )
    generate = TextGeneration(
        name='generate',
        use_system_prompt=True,
        llm = TransformersLLM(
            #model="llama3-8b-8192",
            model_id='meta-llama/Meta-Llama-3-8B-Instruct',
            tokenizer_id='meta-llama/Meta-Llama-3-8B-Instruct',
            model_kwargs={
                'load_in_4bit': True
            },
            structured_output={
                #'format': 'json',
                'schema': SequentialStateMachine.model_json_schema()
            },
        ),
    )
    load_dataset >> set_schema >> generate
    #load_dataset >> generate

if __name__ == '__main__':
    #print(StateMachine.model_json_schema().keys())
    #exit()
    distiset = pipe.run()
    #print(distiset['default']['train'][0]['generation'])
    distiset.push_to_hub('crislmfroes/egpsr-synthetic-data')