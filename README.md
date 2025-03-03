# Next Message Prediction Benchmark

This repository contains code for evaluating language models on the task of predicting the next message in a conversation based on different types of context. It is the code used to generate the dataset and run the experiments in our blog post [Can AI Models Predict What You'll Say Next? Developing Verifiable Social Rewards](https://blog.plasticlabs.ai/research/research/Can-AI-Models-Predict-What-Youll-Say-Next).

## Overview

The benchmark tests a model's ability to identify the genuine next message in a conversation from among three decoys. It's designed to assess social cognition abilities with a verifiable reward signal.

## Key Components

- `generate_dataset.py`: Processes Discord conversations to extract suitable testing samples and generates decoys.
- `run_experiment.py`: Runs a single experiment with a specific model and context mode.
- `run_grid_experiment.py`: Runs multiple experiments across a grid of parameters (models, context modes, etc.)

## Context Modes

The framework supports three types of context:
- **No context**: Only the immediate conversation snippet and options.
- **Raw context**: Includes previous 50 or 100 messages from conversation history.
- **Summary context**: Includes a personality profile generated from the conversation history.

## Usage

To install the dependencies, run:

```bash
uv sync
```

To generate the dataset, run:

```bash
uv run generate_dataset.py --files_dir data/discord-raw --output data/dataset.json
```
Where `data/discord-raw` is a directory containing files with raw Discord data. See the example file `data/discord-raw/channel_a.csv` for the format of the data.

To process the resulting dataset and generate a variant of the dataset that uses less context (both raw and summary), run:

```bash
uv run generate_shorter_context_dataset.py --context_length 10 --input data/dataset_extended_context_with_distractors.json --output_dir data/clean --model claude-3-7-sonnet-20250219 --provider anthropic
```

Where:
- `--context_length`: Number of messages to include in the shortened context (default: 10)
- `--input`: Path to the input dataset file (default: data/dataset_extended_context_with_distractors.json)
- `--output_dir`: Directory to store the new dataset (default: data/clean)
- `--model`: Model to use for regenerating summaries (default: claude-3-7-sonnet-20250219)
- `--provider`: Model provider (default: anthropic)

The output filename will include the context length and model used, e.g., `dataset-10-claude-3-7-sonnet-20250219.json`.

### Running Experiments

To run a single experiment with a specific model and context mode:

```bash
uv run run_experiment.py --dataset DATA_PATH --model MODEL_NAME --context_mode [none|raw|summary] --temperature 0.0 --max-examples 100
```

Where:
- `--dataset`: Path to the dataset file (default: data/dataset.json)
- `--model`: Model to use for evaluation (e.g., claude-3-7-sonnet-20250219, gpt-4o-mini)
- `--context_mode`: How to handle extended context - none, raw, or summary (default: none)
- `--temperature`: Temperature for the model (default: 0.0)
- `--max-examples`: Maximum number of examples to process (default: all)
- `--output`: Path to save the results CSV (default: auto-generated in output/ folder)
- `--random-seed`: Random seed for shuffling options (default: None, uses system time)

Results are saved as CSV files with detailed metrics and as JSON metadata files with summary statistics.

### Running Grid Experiments

To run multiple experiments across a grid of parameters (models, context modes, datasets):

```bash
uv run run_grid_experiment.py --concurrent 4 --datasets data/dataset1.json data/dataset2.json --models gpt-4o claude-3-7-sonnet-20250219 --context_modes none raw summary
```

Where:
- `--concurrent`: Number of experiments to run concurrently (default: 1)
- `--datasets`: List of dataset files to use
- `--models`: List of models to evaluate
- `--context_modes`: List of context modes to test [none|raw|summary]
- `--temperature`: Temperature for model generation (default: 0.0)
- `--max-examples`: Maximum examples to process per experiment (default: all)
- `--output-dir`: Directory to save results (default: auto-generated)

The grid experiment runs all combinations of specified datasets, models, and context modes, providing a comprehensive evaluation across different settings.

### Generating Plots

To visualize experiment results and generate plots for analysis:

```bash
uv run generate_plot.py --results_folders output/experiment1 output/experiment2 --output plots/results.png
```

This script aggregates results from multiple experiment runs, calculates statistics, and generates visualizations comparing model performance across different context types. The plots show accuracy metrics with error bars and are grouped by model family for easier interpretation.

## Experiment Structure

1. Parse Discord data to find suitable conversation snippets
2. Generate three decoy messages for each genuine next message
3. Test models on their ability to identify the genuine message
4. Analyze accuracy across different context modes and models