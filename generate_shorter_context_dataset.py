import json
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar

# Import functions from other files
from generate_dataset import generate_context_summary
from run_experiment import call_model  # Import call_model function

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Generate dataset with shorter context window')
    parser.add_argument('--context_length', type=int, default=10, 
                        help='Number of messages to include in the extended context')
    parser.add_argument('--input', type=str, default="data/dataset_extended_context_with_distractors.json",
                        help='Input dataset file path')
    parser.add_argument('--output_dir', type=str, default="data/clean",
                        help='Output directory for the new dataset')
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-20250219",
                        help='Model to use for generating summaries')
    parser.add_argument('--provider', type=str, default="anthropic",
                        help='Model provider (anthropic, openai, etc.)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output path - include model name in filename
    short_model_name = args.model.split('/')[-1]
    short_model_name = short_model_name.replace('.', '_')
    output_path = os.path.join(args.output_dir, f"dataset-{args.context_length}-{short_model_name}.json")
    
    # Load the original dataset
    with open(args.input, 'r') as f:
        dataset = json.load(f)
    
    # Process each conversation
    processed_dataset = []
    
    # Use tqdm for progress tracking
    for i, conversation in tqdm(enumerate(dataset), total=len(dataset)):
        # Create a copy of the original conversation to modify
        new_conversation = conversation.copy()
        
        # Keep only the last N messages in extended_context
        if 'extended_context' in conversation and len(conversation['extended_context']) > args.context_length:
            new_conversation['extended_context'] = conversation['extended_context'][-args.context_length:]
        
        # Get the target author
        target_author = conversation['target']['Author']
        
        # Regenerate the target_author_profile based on shortened context
        if 'extended_context' in new_conversation and new_conversation['extended_context']:
            # Format the context for the summary
            context_text = ""
            for message in new_conversation['extended_context']:
                context_text += f"{message['Author']}: {message['Content']}\n\n"
            
            # Create the prompt for generating the profile
            prompt = f"""You are analyzing a sequence of messages from a Discord conversation. 
Your task is to create a detailed profile of the user named "{target_author}" based on their messages and interactions.

Focus on:
1. {target_author}'s communication style, personality traits, and apparent expertise
2. Their interests, opinions, and perspectives on topics discussed
3. Their relationship with other participants
4. Any background information that helps understand their thought process
5. Their typical response patterns and how they engage with others

This profile will be used to better understand why {target_author} might respond in certain ways in future messages.

CONVERSATION HISTORY:
{context_text}

Please provide a comprehensive profile of {target_author} based on this conversation history. Focus exclusively on what can be inferred about {target_author}, not the other participants.
"""
            
            response = call_model(
                model=args.model,
                prompt=prompt,
                temperature=0.3
            )
            
            new_conversation['target_author_profile'] = response
            
        else:
            new_conversation['target_author_profile'] = "No extended context available."
        
        processed_dataset.append(new_conversation)
    
    # Save the processed dataset
    with open(output_path, 'w') as f:
        json.dump(processed_dataset, f, indent=2)
    
    print(f"Dataset with context length {args.context_length} using model {args.model} saved to {output_path}")

if __name__ == "__main__":
    main()
