import pandas as pd
import os
import argparse
import json

from anthropic import Anthropic
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

load_dotenv()

def extract_conversations(df: pd.DataFrame, min_length: int = 6, max_length: int = 10) -> list[pd.DataFrame]:
    # First, clean the dataframe by removing rows with NaN content
    df = df.dropna(subset=['Content']).reset_index(drop=True)
    
    conversations: list[pd.DataFrame] = []
    current_conversation: list[int] = []
    
    def is_link_only(content: str) -> bool:
        # Basic check for messages that are just URLs
        return content.strip().startswith('http') and len(content.split()) == 1
    
    def messages_are_related(msg1: pd.Series, msg2: pd.Series) -> bool:
        # Messages within 60 minutes are considered related
        time_diff = pd.to_datetime(msg2['Date']) - pd.to_datetime(msg1['Date'])
        return time_diff.total_seconds() < 3600  # 60 minutes in seconds
    
    def is_valid_conversation(conv_indices: list[int]) -> bool:
        # Check length constraints
        if len(conv_indices) < min_length:
            return False
            
        # Get the conversation slice
        conv_df = df.iloc[conv_indices]
        
        # Check if there are exactly 2 participants
        unique_authors = conv_df['Author'].unique()
        if len(unique_authors) != 2:
            return False
            
        # Check if each participant speaks at least twice
        author_counts = conv_df['Author'].value_counts()
        if any(count < 2 for count in author_counts):
            return False
            
        # Check for at least 2 back-and-forths (A→B→A→B pattern)
        authors = conv_df['Author'].tolist()
        author_a, author_b = unique_authors
        
        # Find first occurrence of each author
        first_a = authors.index(author_a)
        first_b = authors.index(author_b)
        
        # Determine who speaks first
        if first_a < first_b:
            first, second = author_a, author_b
        else:
            first, second = author_b, author_a
            
        # Check for the pattern: first→second→first→second
        pattern_found = False
        for i in range(len(authors) - 3):
            if (authors[i] == first and 
                authors[i+1] == second and 
                authors[i+2] == first and 
                authors[i+3] == second):
                pattern_found = True
                break
                
        return pattern_found
    
    for i in range(len(df)):
        if is_link_only(df.iloc[i]['Content']):
            continue
            
        if not current_conversation:
            current_conversation.append(i)
            continue
            
        if messages_are_related(df.iloc[current_conversation[-1]], df.iloc[i]):
            current_conversation.append(i)
            
            # Check if we've reached max length
            if len(current_conversation) == max_length:
                # If the conversation is valid, save it
                if is_valid_conversation(current_conversation):
                    conversations.append(df.iloc[current_conversation].copy())
                current_conversation = []  # Start a new conversation
        else:
            # If the conversation is valid, save it
            if is_valid_conversation(current_conversation):
                conversations.append(df.iloc[current_conversation].copy())
            current_conversation = [i]  # Start a new conversation with this message
    
    # Add the last conversation if it's valid
    if current_conversation and is_valid_conversation(current_conversation):
        conversations.append(df.iloc[current_conversation].copy())
    
    return conversations


def save_conversations(formatted_data: list[dict], output_file: str) -> None:
    import json
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)


def get_extended_context(df: pd.DataFrame, current_indices: list[int], window_size: int = 100) -> pd.DataFrame:
    """
    Get an extended context window around the current conversation.
    
    Args:
        df: The full dataframe of messages
        current_indices: The indices of the current conversation
        window_size: How many messages to include in the extended context
    
    Returns:
        DataFrame containing the extended context messages
    """
    # Get the earliest message index in our current conversation
    earliest_idx = min(current_indices)
    
    # Calculate start and end indices for the extended context
    # We want to get messages before our conversation started
    start_idx = max(0, earliest_idx - window_size // 2)
    end_idx = min(len(df), earliest_idx + window_size // 2)
    
    # Get the extended context
    extended_context_indices = list(range(start_idx, end_idx))
    return df.iloc[extended_context_indices].copy()


@observe(as_type="generation")
def generate_context_summary(extended_context: pd.DataFrame,
                             target_author: str,
                             client: Anthropic,
                             model: str = "claude-3-7-sonnet-20250219") -> str:

    """
    Generate a summary of the extended context focused on the target author.
    
    Args:
        extended_context: DataFrame containing the extended context messages
        target_author: The username of the author who wrote the target message
        client: The API client
        model: The model to use
    
    Returns:
        A summary focused on the target author
    """
    # Format the extended context for the LLM
    context_text = ""
    for _, row in extended_context.iterrows():
        context_text += f"{row['Author']}: {row['Content']}\n\n"
    
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

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.3,  # Lower temperature for more factual summary
        system=f"You are an expert at analyzing conversation patterns and creating user profiles. Focus exclusively on the user {target_author}.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract the summary text
    # Handle different content block types that might be returned
    if hasattr(response.content[0], 'text'):
        summary = response.content[0].text
    else:
        # For ToolUseBlock, ThinkingBlock, etc., access the content differently
        summary = str(response.content[0])
    
    # See docs for more details on token counts and usd cost in Langfuse
    # https://langfuse.com/docs/model-usage-and-cost
    langfuse_context.update_current_observation(
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
    )
    print(f'Input tokens: {response.usage.input_tokens}')
    print(f'Output tokens: {response.usage.output_tokens}')
    return summary


def format_conversations_with_extended_context(conversations: list[pd.DataFrame], 
                                              df: pd.DataFrame, 
                                              client,
                                              window_size: int = 100) -> list[dict]:
    """
    Format conversations and add extended context summaries focused on the target author.
    
    Args:
        conversations: List of conversation DataFrames
        df: The full dataframe of messages
        client: The API client
    
    Returns:
        Formatted conversation data with extended context
    """
    formatted_data: list[dict] = []
    
    for conv in conversations:
        # Get indices of this conversation in the original dataframe
        indices = conv.index.tolist()
        
        # Get the target author (who wrote the last message)
        target_author = conv.iloc[-1]['Author']
        
        # Get extended context
        extended_context = get_extended_context(df, indices, window_size=window_size)
        
        # Generate summary of extended context focused on target author
        context_summary = generate_context_summary(extended_context, target_author, client)
        
        # Format the conversation as before
        context = conv.iloc[:-1][['Author', 'Content']].to_dict('records')
        target = conv.iloc[-1][['Author', 'Content']].to_dict()
        
        formatted_data.append({
            'context': context,
            'target': target,
            'distractors': [],
            'target_author_profile': context_summary  # Renamed to be more specific
        })
    
    return formatted_data

    
def process_files(files: list[str],
                  client: Anthropic,
                  output_file: str = '',
                  data_folder: str = 'data/lab-raw',
                  window_size: int = 100,
                  model: str = "claude-3-7-sonnet-20250219") -> list[dict]:
    """
    Process multiple Discord log files to extract conversations with extended context.
    
    Args:
        files: List of filenames to process
        client: The API client for LLM inference
        output_file: If provided, save the formatted data to this file
        data_folder: Directory containing the input files
        
    Returns:
        The formatted conversation data
    """
    # Load all dataframes into one and sort by date
    all_dfs = []
    for file in files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)
        # Add channel name as a column
        df['Channel'] = file.replace('.csv', '')
        all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert Date column to datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    # Process each file separately to extract conversations
    all_formatted_data = []
    
    for file in files:
        channel = file.replace('.csv', '')
        print(f"Processing channel: {channel}")
        
        # Get dataframe for this channel
        channel_df = pd.read_csv(os.path.join(data_folder, file))
        channel_df = channel_df.dropna(subset=['Content']).reset_index(drop=True)
        channel_df['Date'] = pd.to_datetime(channel_df['Date'])
        
        # Extract conversations using existing logic
        conversations = extract_conversations(channel_df, min_length=6, max_length=10)
        print(f"Found {len(conversations)} conversations in {channel}")
        
        # Process each conversation
        for conv in conversations:
            # Get the target author (who wrote the last message)
            target_author = conv.iloc[-1]['Author']
            target_date = conv.iloc[-1]['Date']
            
            # Get extended context: last 50 messages across all channels within 48 hours before the target message
            time_threshold = target_date - pd.Timedelta(hours=48)
            extended_context = combined_df[
                (combined_df['Date'] < target_date) & 
                (combined_df['Date'] >= time_threshold)
            ].tail(50)
            
            # If we have extended context, generate a summary
            if len(extended_context) > 0:
                # Generate summary of extended context focused on target author
                context_summary = generate_context_summary(extended_context, target_author, client, model=model)
            else:
                context_summary = "No extended context available."
            
            # Format date
            conv_copy = conv.copy()
            conv_copy['Date'] = conv_copy['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            extended_context_copy = extended_context.copy()
            extended_context_copy['Date'] = extended_context_copy['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

            # Format the conversation
            context = conv_copy.iloc[:-1][['Author', 'Content', 'Date']].to_dict('records')
            target = conv_copy.iloc[-1][['Author', 'Content', 'Date']].to_dict()

            # Format the extended context
            extended_context_records = extended_context_copy[['Author', 'Content', 'Date', 'Channel']].to_dict('records')
            
            # Create the formatted data entry
            formatted_data = {
                'extended_context': extended_context_records,
                'context': context,
                'target': target,
                'distractors': [],
                'target_author_profile': context_summary,
                'channel': channel
            }
            
            all_formatted_data.append(formatted_data)
    
    print(f"Total conversations extracted: {len(all_formatted_data)}")
    
    if output_file:
        save_conversations(all_formatted_data, output_file)
    
    return all_formatted_data

    
def create_distractor_prompt(conversation: dict) -> str:
    conversation_str = ''
    for message in conversation['context']:
        conversation_str += f"{message['Author']}: {message['Content']}\n"
    conversation_str += f"Author: {conversation['target']['Author']}\n"
    conversation_str += f"Content: {conversation['target']['Content']}\n"
    
    prompt = f"""You are helping create a dataset for evaluating language models on conversation understanding. 

    Given a conversation context and the actual next message that was sent, your task is to generate 3 alternative messages that:
    1. Could plausibly be the next message in the conversation
    2. Are written in the same style as the original author
    3. Are meaningfully different from the actual message
    4. Maintain the same general topic but change the specific content, opinion, or direction
    5. Are approximately the same length as the actual message
    6. Are not just a rephrasing of the actual message

    <conversation>
    {conversation_str}
    </conversation>

    <target>
    {conversation['target']['Author']}: {conversation['target']['Content']}
    </target>

    Generate 3 alternative messages that {conversation['target']['Author']} might have plausibly written instead. Make them convincing but distinctly different from the actual message. Each should be a complete, standalone message.

    Output them in valid JSON and nothing else, using the following format:
    {{
        "distractors": ["message 1", "message 2", "message 3"]
    }}
    """
    return prompt


def generate_distractors(conversation: dict, client: Anthropic, model: str = "claude-3-7-sonnet-20250219") -> list[str]:
    prompt = create_distractor_prompt(conversation)
    response = client.messages.create(
        model=model,
        temperature=0.2,
        max_tokens=10000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract the content safely
    if hasattr(response.content[0], 'text'):
        content_text = response.content[0].text
    else:
        content_text = str(response.content[0])
        
    return json.loads(content_text)['distractors']


def get_dataset_distractors(
    dataset: list[dict], 
    client: Anthropic, 
    model: str = "claude-3-7-sonnet-20250219", 
    max_retries: int = 3
) -> list[dict]:
    """
    Generate distractors for all conversations in the dataset.
    
    Args:
        dataset: List of conversation data
        client: The API client
        model: The model to use
        max_retries: Maximum number of retry attempts for failed generations
        
    Returns:
        Dataset with distractors added
    """
    print(f"Generating distractors for {len(dataset)} conversations...")
    dataset_with_distractors = []
    
    # Track conversations that need distractor generation
    remaining_conversations = list(enumerate(dataset))
    retry_count = 0
    
    # Continue until all conversations have distractors or max retries exceeded
    while remaining_conversations and retry_count < max_retries:
        if retry_count > 0:
            print(f"Retry attempt {retry_count} for {len(remaining_conversations)} conversations")
        
        next_remaining = []
        
        for idx, conversation in remaining_conversations:
            try:
                print(f"Processing conversation {idx+1}/{len(dataset)}")
                distractors = generate_distractors(conversation, client, model=model)
                conversation_copy = conversation.copy()
                conversation_copy['distractors'] = distractors
                dataset_with_distractors.append(conversation_copy)
                print(f"Successfully generated distractors for conversation {idx+1}")
            except Exception as e:
                print(f"Error generating distractors for conversation {idx+1}: {e}")
                next_remaining.append((idx, conversation))
        
        remaining_conversations = next_remaining
        retry_count += 1
    
    if remaining_conversations:
        print(f"Warning: Could not generate distractors for {len(remaining_conversations)} conversations after {max_retries} retries")
    
    return dataset_with_distractors


def main():
    parser = argparse.ArgumentParser(description='Generate conversation dataset with extended context')
    parser.add_argument('--files_dir', type=str, default="data/lab-raw", 
                        help='Directory containing the CSV files to process')
    parser.add_argument('--output', type=str, default="data/dataset_extended_context.json",
                        help='Output JSON file path')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Size of extended context window (number of messages)')
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-20250219",
                        help='Anthropic model to use')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum number of retries for generating distractors')
    parser.add_argument('--skip_distractors', action='store_true',
                        help='Skip generating distractors')
    
    args = parser.parse_args()
    
    # Get all CSV files in the directory
    files = [f for f in os.listdir(args.files_dir) if f.endswith('.csv')]
    if not files:
        print(f"No CSV files found in {args.files_dir}")
        return
    
    print(f"Found {len(files)} CSV files to process: {', '.join(files)}")
    
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    dataset = process_files(files=files, client=client, output_file=args.output, 
                           data_folder=args.files_dir, window_size=args.window_size, model=args.model)

    # Generate distractors if not skipped
    if not args.skip_distractors:
        dataset_with_distractors = get_dataset_distractors(
            dataset=dataset, 
            client=client, 
            model=args.model,
            max_retries=args.max_retries
        )
        
        # Save final dataset with distractors
        if dataset_with_distractors:
            distractor_output = args.output.replace('.json', '_with_distractors.json')
            print(f"Saving {len(dataset_with_distractors)} conversations with distractors to {distractor_output}")
            with open(distractor_output, 'w') as f:
                json.dump(dataset_with_distractors, f, indent=2)


if __name__ == "__main__":
    main()
