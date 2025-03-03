#!/usr/bin/env python3
"""
Multiple-choice experiment for language models on predicting next messages in conversations.

This script loads a dataset of conversations with targets and distractors,
formats them as multiple-choice questions, sends them to a language model,
and evaluates the results.
"""

import json
import random
import os
import time
import argparse
import string
import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import google.generativeai as genai

load_dotenv()

ANTHROPIC_MAX_THINKING_TOKENS = 2000

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a multiple-choice experiment using language models"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/dataset.json", 
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="claude-3-7-sonnet-20250219", 
        help="Model to use (e.g., claude-3-7-sonnet-20250219, gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to save the results CSV (default: auto-generated in output/ folder)"
    )
    parser.add_argument(
        "--max-examples", 
        type=int, 
        default=None, 
        help="Maximum number of examples to process (default: all)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Temperature for the model (default: 0.0)"
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        choices=["none", "raw", "summary"],
        default="none",
        help="How to handle extended context: none, raw, or summary (default: none)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for shuffling options (default: None, uses system time)"
    )
    return parser.parse_args()

def generate_output_filename(args: argparse.Namespace, provider: str) -> Tuple[str, str]:
    """Generate output filenames based on input parameters."""
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Shorten model name for filename
    model_name = args.model.replace("/", "-").replace(".", "-")
    
    # Create base filename
    base_filename = f"{model_name}_{args.context_mode}_{timestamp}"
    
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate paths
    csv_path = output_dir / f"{base_filename}.csv"
    json_path = output_dir / f"{base_filename}_metadata.json"
    
    return str(csv_path), str(json_path)

def save_metadata(args: argparse.Namespace, json_path: str, csv_path: str, accuracy: float, channel_acc: pd.DataFrame) -> None:
    """Save metadata including input parameters and results summary to JSON file."""
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Convert provider to string if None for JSON serialization
    provider = args.provider or get_provider_from_model(args.model)
    
    # Convert channel accuracy to serializable format
    channel_results = {}
    for channel, row in channel_acc.iterrows():
        channel_results[channel] = {
            "accuracy": float(row["mean"]),
            "count": int(row["count"])
        }
    
    metadata = {
        "parameters": {
            **args_dict,
            "provider": provider
        },
        "results": {
            "output_csv": csv_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_accuracy": float(accuracy),
            "channel_accuracy": channel_results,
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {json_path}")

def load_dataset(file_path: str) -> List[Dict]:
    """Load the dataset from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def format_message(message: Dict) -> str:
    """Format a single message for display."""
    return f"{message['Author']}: {message['Content']}"

def format_context(context: List[Dict], context_mode: str = "none", extended_context: Optional[List[Dict]] = None, target_author_profile: Optional[str] = None) -> str:
    """Format the context messages based on the specified mode."""
    # Format the standard context messages (immediate history)
    immediate_context = "\n".join(format_message(msg) for msg in context)
    
    if context_mode == "none" or (extended_context is None and target_author_profile is None):
        return immediate_context
    
    # Add extended context if requested
    if context_mode == "raw" and extended_context:
        # Add the raw extended context before the immediate context
        raw_extended = "\n".join(format_message(msg) for msg in extended_context)
        return f"# Extended Previous History\n{raw_extended}\n\n# Conversation\n{immediate_context}"
    
    # Add author profile summary if requested
    if context_mode == "summary" and target_author_profile:
        return f"# Target Author Profile\n{target_author_profile}\n\n# Immediate Conversation\n{immediate_context}"
    
    # Default fallback
    return immediate_context

def format_options(target: Dict, distractors: List, random_seed: Optional[int] = None) -> Tuple[List[str], Dict[str, Any]]:
    """Format the options (target + distractors) as multiple choice options."""
    # Set random seed if provided for reproducible shuffling
    if random_seed is not None:
        random.seed(random_seed)
        
    # Convert string distractors to dict format if needed
    formatted_distractors = []
    for distractor in distractors:
        if isinstance(distractor, str):
            # Create a dictionary similar to the target but with different content
            formatted_distractors.append({
                "Author": target["Author"],
                "Content": distractor,
                "Date": target["Date"]
            })
        else:
            formatted_distractors.append(distractor)
    
    # Combine target and distractors
    options = [target] + formatted_distractors
    random.shuffle(options)  # Shuffle to avoid position bias
    
    # Create the options mapping
    option_letters = list(string.ascii_uppercase)[:len(options)]
    option_mapping = {}
    formatted_options = []
    
    for letter, option in zip(option_letters, options):
        formatted_options.append(f"{letter}. {format_message(option)}")
        option_mapping[letter] = option
    
    return formatted_options, option_mapping

def create_prompt(context: List[Dict], target: Dict, distractors: List, context_mode: str = "none", extended_context: Optional[List[Dict]] = None, target_author_profile: Optional[str] = None, random_seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
    """Create the prompt for the language model."""
    context_str = format_context(
        context, 
        context_mode=context_mode,
        extended_context=extended_context,
        target_author_profile=target_author_profile
    )
    formatted_options, option_mapping = format_options(target, distractors, random_seed=random_seed)
    
    # Create introduction text based on context mode
    intro_text = "Given the following conversation history, which message do you think was sent next?"
    
    if context_mode == "raw":
        intro_text = """You'll see two sections below:
1. First, some older messages from the same conversation/channel
2. Then, the most recent messages right before the next response

Based on both the extended history and the recent conversation, predict which message was sent next."""
    
    elif context_mode == "summary":
        intro_text = """You'll see two sections below:
1. First, a profile summarizing the communication style and background of the target author
2. Then, the most recent messages in a conversation

Using both the author's profile and the recent conversation context, predict which message this author sent next."""
    
    # Create the prompt with appropriate introduction
    prompt = f"""{intro_text}

<conversation>
{context_str}
</conversation>

Choose the most likely next message from these options:
<options>
{os.linesep.join(formatted_options)}
</options>

Respond only with the letter of the option you think was sent next, and nothing else.
"""
    return prompt, option_mapping

def get_provider_from_model(model: str) -> str:
    """Determine the provider based on the model name."""
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower or "o1" in model_lower or "o3-" in model_lower:
        return "openai"
    elif "llama" in model_lower or "hermes" in model_lower or "deepseek" in model_lower:
        return "openrouter"
    elif "gemini" in model_lower:
        return "google"
    else:
        raise ValueError(f"Could not determine provider from model name: {model}")

def call_model(prompt: str, model: str, temperature: float = 0.0, provider: Optional[str] = None) -> Optional[str]:
    """Call the appropriate API based on the provider."""
    # Determine provider if not provided
    if provider is None:
        provider = get_provider_from_model(model)
        
    if provider == "anthropic":
        return call_anthropic(prompt, model, temperature)
    elif provider == "openai":
        return call_openai(prompt, model, temperature)
    elif provider == "openrouter":
        return call_openai(prompt, model, temperature, base_url="https://openrouter.ai/api/v1")
    elif provider == "google":
        return call_gemini(prompt, model, temperature)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def call_anthropic(prompt: str, model: str, temperature: float = 0.0) -> Optional[str]:
    """Call the Anthropic API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if model.endswith("-thinking"):
        thinking = True
        model = model.replace("-thinking", "")
    else:
        thinking = False
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    
    client = Anthropic(api_key=api_key)
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        if thinking:
            kwargs["temperature"] = 1
            kwargs["max_tokens"] = ANTHROPIC_MAX_THINKING_TOKENS + 500
            kwargs["thinking"] = {
                "budget_tokens": ANTHROPIC_MAX_THINKING_TOKENS,
                "type": "enabled"
            }
        else:
            kwargs["max_tokens"] = 500
            kwargs["temperature"] = temperature

        message = client.messages.create(**kwargs)

        if message.content and len(message.content) > 0:
            if thinking:
                content_block = message.content[1]
            else:
                content_block = message.content[0]
            if hasattr(content_block, 'text'):
                response = content_block.text.strip()
                return response
        return ""
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        time.sleep(1)  # Wait a bit before retrying
        return None
    
def call_gemini(prompt: str, model: str, temperature: float = 0.0) -> Optional[str]:
    """Call the Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model, generation_config=genai.GenerationConfig(temperature=temperature)) # type: ignore
        response = model.generate_content(prompt)
        print(f'response: {response}')
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        time.sleep(1)  # Wait a bit before retrying
        return None

def call_openai(prompt: str, model: str, temperature: float = 0.0, base_url: Optional[str] = None) -> Optional[str]:
    """Call the OpenAI API or compatible API like OpenRouter."""
    # Determine which API key to use
    if base_url and "openrouter.ai" in base_url:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Configure client with base_url if provided
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    # Add OpenRouter specific headers if needed
    if base_url and "openrouter.ai" in base_url:
        headers = {
            "HTTP-Referer": "https://your-app-url.com",  # Required by OpenRouter
            "X-Title": "AuthorStyle Experiment"  # Optional identifier
        }
        client_kwargs["default_headers"] = headers
    
    client = OpenAI(**client_kwargs)
    
    # Common parameters
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if "o1" in model or "o3-" in model:
        params["max_completion_tokens"] = 1200
        params["reasoning_effort"] = "medium"
    else:
        params["max_tokens"] = 500
        params["temperature"] = temperature

    # if "openrouter" in base_url:
    #     params['provider'] = {'ignore': ['inference.net']}

    try:
        response = client.chat.completions.create(**params)
        # print(f'Finish reason: {response.choices[0].finish_reason}')
        # print(f'response: {response.choices[0].message.content}')
        response_text = response.choices[0].message.content.strip()
        # print(f'prompt: {prompt}')
        # print(f'choice: {response_dict["answer"]} explanation: {response_dict["explanation"]}')
        return response_text
    except Exception as e:
        print(f"Error calling API: {e}")
        time.sleep(1)  # Wait a bit before retrying
        return None
    

def extract_answer(response: str) -> Optional[str]:
    """Extract the answer letter from the model's response."""
    if not response:
        return None
    
    # Try to find any uppercase letter that might be the answer
    for char in response:
        if char in string.ascii_uppercase:
            return char
    
    # If that fails, try lowercase
    for char in response:
        if char in string.ascii_lowercase:
            return char.upper()
    
    return None

def evaluate_example(example: Dict, model: str, temperature: float = 0.0, context_mode: str = "none", random_seed: Optional[int] = None) -> Dict:
    """Evaluate a single example."""
    # Get provider for the model
    provider = get_provider_from_model(model)
    
    context = example["context"]
    target = example["target"]
    distractors = example["distractors"]
    
    # Get extended context or author profile if available and requested
    extended_context = example.get("extended_context") if context_mode == "raw" else None
    target_author_profile = example.get("target_author_profile") if context_mode == "summary" else None
    
    prompt, option_mapping = create_prompt(
        context, 
        target, 
        distractors,
        context_mode=context_mode,
        extended_context=extended_context,
        target_author_profile=target_author_profile,
        random_seed=random_seed
    )
    
    # Call the model
    response = call_model(prompt, model, temperature)
    answer = extract_answer(response)
    # Check if the answer is correct
    selected_option = option_mapping.get(answer) if answer else None
    is_correct = (selected_option == target) if selected_option else False
    # print(f'answer: {answer} selected_option: {selected_option} target: {target} is_correct: {is_correct}')
    
    # Find the correct letter
    correct_letter = None
    for letter, option in option_mapping.items():
        if option == target:
            correct_letter = letter
            break
    
    return {
        "prompt": prompt,
        "response": response,
        "answer": answer,
        "correct_letter": correct_letter,
        "is_correct": is_correct,
        "channel": example.get("channel", "unknown"),
        "context_mode": context_mode,
        "model": model,
        "provider": provider
    }

def run_experiment(
    dataset_path: str, 
    model: str, 
    output_path: Optional[str] = None, 
    max_examples: Optional[int] = None, 
    temperature: float = 0.0,
    context_mode: str = "none",
    random_seed: Optional[int] = None
):
    """Run the experiment on the dataset."""
    # Load the dataset
    dataset = load_dataset(dataset_path)
    
    # Limit the number of examples if specified
    if max_examples is not None:
        dataset = dataset[:max_examples]
    
    # Get provider for the model
    provider = get_provider_from_model(model)
    
    # Generate output filenames if not provided
    args = argparse.Namespace(
        dataset=dataset_path,
        model=model,
        output=output_path,
        max_examples=max_examples,
        temperature=temperature,
        context_mode=context_mode,
        random_seed=random_seed,
        provider=provider
    )
    
    if output_path is None:
        csv_path, json_path = generate_output_filename(args, provider)
    else:
        csv_path = output_path
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Generate JSON path from CSV path
        json_path = os.path.splitext(csv_path)[0] + "_metadata.json"
    
    results = []
    
    # Process each example
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing examples"):
        # If random_seed is provided, create a deterministic per-example seed
        example_seed = random_seed + i if random_seed is not None else None
        
        result = evaluate_example(
            example, 
            model, 
            temperature, 
            context_mode, 
            random_seed=example_seed
        )
        
        # Add metadata
        result["example_id"] = i
        result["context_length"] = len(example["context"])
        result["num_options"] = len(example["distractors"]) + 1  # target + distractors
        
        results.append(result)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Calculate overall accuracy
    accuracy = df["is_correct"].mean()
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # Calculate accuracy by channel
    channel_acc = df.groupby("channel")["is_correct"].agg(["mean", "count"])
    print("\nAccuracy by channel:")
    print(channel_acc)
    
    # Save results to CSV
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Save metadata to JSON
    save_metadata(args, json_path, csv_path, accuracy, channel_acc)
    
    return df

def main():
    """Main entry point."""
    args = parse_args()
    run_experiment(
        dataset_path=args.dataset,
        model=args.model,
        output_path=args.output,
        max_examples=args.max_examples,
        temperature=args.temperature,
        context_mode=args.context_mode,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main() 