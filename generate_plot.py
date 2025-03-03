import pandas as pd
import os
import json

results_folders = [
    'output/FINAL-grid_experiment_20250227_194539',
    'output/FINAL-grid_experiment_20250227_230612',
    'output/FINAL-grid_experiment_20250227_234520',
    'output/FINAL-grid_experiment_20250228_000836',
    'output/FINAL-grid_experiment_20250228_114923',
    'output/grid_experiment_20250228_114847'
]
                   
results_files = []
for results_folder in results_folders:
    folder_files = os.listdir(results_folder)
    folder_files = [f for f in folder_files if f.endswith('.json')]
    # add folder name to each file
    folder_files = [f'{results_folder}/{f}' for f in folder_files]
    results_files.extend(folder_files)
results = []
for file in results_files:
    results_json = json.load(open(file))
    results_dict = {}
    results_dict['model'] = results_json['parameters']['model']
    results_dict['dataset'] = results_json['parameters']['dataset']
    results_dict['n_context'] = map
    results_dict['context_mode'] = results_json['parameters']['context_mode']
    results_dict['accuracy'] = results_json['results']['overall_accuracy']
    results_dict['random_seed'] = results_json['parameters']['random_seed']
    results.append(results_dict)
results_df = pd.DataFrame(results)
results_df['n_context'] = results_df['dataset'].str.extract(r'dataset-(\d+)-')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Prepare data
results_df['n_context'] = pd.to_numeric(results_df['n_context'], errors='coerce')

# Create a combined context type field
def get_context_type(row):
    if row['context_mode'] == 'none':
        return 'No Context'
    else:
        return f"{int(row['n_context'])} {row['context_mode'].capitalize()}"

results_df['context_type'] = results_df.apply(get_context_type, axis=1)

# Calculate mean, std, and count for each model and context type
grouped_stats = results_df.groupby(['model', 'context_type']).agg(
    mean_accuracy=('accuracy', 'mean'),
    std=('accuracy', 'std'),
    count=('accuracy', 'count')
).reset_index()

# Calculate standard error
grouped_stats['stderr'] = grouped_stats['std'] / np.sqrt(grouped_stats['count'])

# Define model families for grouping
model_families = {
    'Claude': ['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-latest'],
    'OpenAI': ['gpt-4.5-preview-2025-02-27', 'o1', 'o3-mini', 'gpt-4o', 'gpt-4o-mini'],
    'Google': ['gemini-2.0-flash'],
    'Llama': ['meta-llama/llama-3.3-70b-instruct'],
    'Hermes': ['nousresearch/hermes-3-llama-3.1-405b', 'nousresearch/hermes-3-llama-3.1-70b'],
    'DeepSeek': ['deepseek/deepseek-chat', 'deepseek/deepseek-r1']
}

# Create model family column
def get_model_family(model):
    for family, models in model_families.items():
        if model in models:
            return family
    return "Other"

grouped_stats['model_family'] = grouped_stats['model'].apply(get_model_family)

# Shortened model names for display
model_display_names = {
    'claude-3-7-sonnet-20250219': 'Claude 3.7',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
    'claude-3-5-haiku-latest': 'Claude 3.5 Haiku',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4.5-preview-2025-02-27': 'GPT-4.5',
    'o3-mini': 'O-3 Mini',
    'o1': 'O-1',
    'deepseek/deepseek-chat': 'V3',
    'deepseek/deepseek-r1': 'R1',
    'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'nousresearch/hermes-3-llama-3.1-405b': 'Hermes 405B',
    'nousresearch/hermes-3-llama-3.1-70b': 'Hermes 70B',
    'gemini-2.0-flash': 'Gemini 2.0 Flash'
}
grouped_stats['model_display'] = grouped_stats['model'].map(model_display_names)

# Set up colors for context modes
colors = {
    'No Context': '#1f77b4',    # Blue
    '50 Raw': '#ff7f0e',        # Orange
    '50 Summary': '#2ca02c',    # Green
    '100 Raw': '#d62728',       # Red
    '100 Summary': '#9467bd'    # Purple
}

# Set up figure
fig, ax = plt.subplots(figsize=(18, 10))

# Create positions for each model group
family_order = ['Claude', 'OpenAI', 'Google', 'Llama', 'Hermes', 'DeepSeek']
model_order = []
for family in family_order:
    for model in model_families[family]:
        if model in grouped_stats['model'].values:
            model_order.append(model)

# Plot each context mode
bar_width = 0.15
x = np.arange(len(model_order))

# Store bar objects for legend
bar_objects = []

# Order of context types
context_types = ['No Context', '50 Raw', '50 Summary', '100 Raw', '100 Summary']

# Main plotting loop
for i, context_type in enumerate(context_types):
    values = []
    errors = []
    
    for model in model_order:
        data = grouped_stats[(grouped_stats['model'] == model) & 
                            (grouped_stats['context_type'] == context_type)]
        if not data.empty:
            values.append(data['mean_accuracy'].values[0])
            errors.append(data['stderr'].values[0])
        else:
            values.append(0)
            errors.append(0)
    
    # Calculate position offset
    offset = (i - 2) * bar_width
    
    # Store the bar object for legend
    bars = ax.bar(x + offset, values, bar_width, 
                color=colors[context_type], alpha=0.8, 
                yerr=errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})
    
    bar_objects.append(bars)

# Add horizontal line at 25% for random guessing
random_line = plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.7)

# Add text box for random guess line
plt.text(len(model_order)-1, 0.26, 'Random Guess (25%)', 
         fontsize=10, color='red', ha='right', va='bottom',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

# Add vertical separators between model families
separator_positions = []
current_family = None
for i, model in enumerate(model_order):
    family = get_model_family(model)
    if family != current_family and i > 0:
        separator_positions.append(i - 0.5)
    current_family = family

for pos in separator_positions:
    plt.axvline(x=pos, color='black', linestyle='--', alpha=0.5)

# Add labels
label = ax.set_xlabel('Model', fontweight='bold', fontsize=14)
label.set_position((0.5, -0.3))  # (x, y) position relative to bottom of axis

plt.title('Model Performance by Context Mode', fontweight='bold', fontsize=16)
plt.xticks(x, [model_display_names.get(model, model) for model in model_order], rotation=45, ha='right')

# Create legend
plt.legend(bar_objects, context_types, title='Context Mode', loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.3)

# Set y-axis to 0-100%
plt.ylim(0, 1.0)
plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(val*100)}%' for val in np.arange(0, 1.1, 0.1)])

# Calculate the center positions for each family
family_centers = {}
family_indices = {}

# First, collect indices for each family
for i, model in enumerate(model_order):
    family = get_model_family(model)
    if family not in family_indices:
        family_indices[family] = []
    family_indices[family].append(i)

# Then calculate center position for each family
for family, indices in family_indices.items():
    family_centers[family] = np.mean(indices)

# Add centered model family labels
for family, center in family_centers.items():
    ax.annotate(family, xy=(center, 0.7), xycoords=('data', 'axes fraction'), 
                ha='center', va='top', fontsize=12, alpha=0.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Increased bottom margin for labels
# plt.show()

plt.savefig('model_performance_by_context_mode.png', dpi=300)