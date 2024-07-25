# How It Works

A powerfull frontier model, such as `Llama3.1-405B` or `Mistral Large` is feed with instructions from the RoboCup@Home command generator, stored in the `instruction` column of the input dataset, and prompted to answer with a state machine that will allow a service robot to execute the command. The `outlines` integration of `distilabel` is used in order to enforce a consistent structured output format for the state machine, as well as an in-context example in the model's prompt. The generated JSON is then added to the output dataset as the column `generation`.

# Install

```sh
pip install -r requirements.txt
```

# Configure generator

Edit `config.py` with the desired service robot instructions dataset, model to be used for synthetic data generation, and output dataset to be generated with reasoning traces and task state machines.

# Run generator

```sh
python process.py
```