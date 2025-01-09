import argparse
import os
import openai
import sys
import json
import textwrap

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate species descriptions, diagnoses, and a dichotomous key using GPT-4.")
    parser.add_argument('--description_file', type=str, required=True, help='Path to the description text file.')
    parser.add_argument('--example_file', type=str, required=True, help='Path to the example text file serving as a writing style template, including an example dichotomous key.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the generated outputs (both .txt and .tex formats).')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for the output files (default: output).')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature to use (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=30000, help='Maximum number of tokens to generate in the output (default: 2000).')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for inference (default: gpt-4o).')
    return parser.parse_args()

def read_description_file(file_path):
    """
    Reads the description file and splits it into separate entries for each species.
    Assumes that each species entry starts with 'Species Description for [Species Name]:'
    and is separated by a line with '#' characters.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Description file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split content into species entries based on the delimiter line
    species_entries = content.strip().split('#' * 80)
    species_data_list = []
    for entry in species_entries:
        entry = entry.strip()
        if entry:
            species_data_list.append(entry)
    return species_data_list

def read_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_species_name(species_data):
    """
    Extracts the species name from the species data.
    Assumes that the species data starts with 'Species Description for [Species Name]:'
    """
    lines = species_data.strip().split('\n')
    for line in lines:
        if line.startswith('Species Description for'):
            # Extract the part after 'Species Description for' and before ':'
            parts = line.split('Species Description for', 1)
            if len(parts) > 1:
                species_name_part = parts[1].strip()
                species_name = species_name_part.rstrip(':').strip()
                return species_name
    return 'unknown_species'

def extract_qa_section(species_data):
    """
    Extracts the QA section from the species data.
    Assumes that the QA section starts after 'QA Section:'.
    """
    lines = species_data.strip().split('\n')
    qa_lines = []
    recording = False
    for line in lines:
        if line.strip() == 'QA Section:':
            recording = True
            continue
        if recording:
            if line.strip() == '':
                break  # Stop if there's an empty line
            qa_lines.append(line)
    return '\n'.join(qa_lines)

# --- Modified extract_measurements function ---
def extract_measurements(species_data):
    """
    Extracts the measurements from the species data, excluding major and minor axis.
    Assigns measurements to 'Uncategorized' if no category is set.
    """
    lines = species_data.strip().split('\n')
    measurements = {}
    current_category = None  # Changed from '' to None
    recording = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == 'Measurements:':
            recording = True
            continue
        if recording:
            if stripped_line == '':
                break  # Stop if there's an empty line
            if stripped_line.startswith('- **') and stripped_line.endswith('**:'):
                # It's a new category
                category = stripped_line[4:-3].strip()
                current_category = category
                measurements[current_category] = []
            else:
                if 'Major Axis' in stripped_line or 'Minor Axis' in stripped_line:
                    continue  # Exclude major and minor axis
                if current_category is None:
                    # Assign to a default category if none is set
                    current_category = 'Uncategorized'
                    measurements[current_category] = []
                measurements[current_category].append(stripped_line)
    return measurements

def extract_materials_examined(species_data):
    """
    Extracts the Materials Examined section from the species data.
    Assumes that the section starts after 'Materials Examined:'.
    """
    lines = species_data.strip().split('\n')
    materials_lines = []
    recording = False
    for line in lines:
        if line.strip() == 'Materials Examined:':
            recording = True
            continue
        if recording:
            if line.strip() == '':
                break  # Stop if there's an empty line
            materials_lines.append(line)
    return '\n'.join(materials_lines)

def format_measurements(measurements):
    """
    Formats the measurements with line wrapping.
    """
    formatted = "Morphological Measurements:\n"
    for category, data_list in measurements.items():
        data_str = ' '.join(data_list)
        wrapped_data = textwrap.fill(data_str, width=80, subsequent_indent='    ')
        formatted += f"- **{category}**: {wrapped_data}\n"
    return formatted

def format_materials_examined(materials):
    """
    Formats the Materials Examined section with line wrapping.
    """
    formatted = "Materials Examined:\n"
    wrapped_materials = textwrap.fill(materials, width=80, subsequent_indent='    ')
    formatted += wrapped_materials
    return formatted

def construct_description_prompt(qa_descriptions, example_text):
    """
    Constructs the prompt for generating species descriptions.
    """
    combined_qa_descriptions = "\n\n".join(qa_descriptions)
    prompt = f"""You are provided with QA descriptions for multiple species and an example of the desired writing style.

**Instructions:**

1. **Do not fabricate or make up any information.** Use only the data provided in the QA descriptions.

2. **For each species**, generate a **Species Description** based on the QA section, including coloration and morphological features.

3. **Follow the style and format** of the provided example.

**QA Descriptions:**

{combined_qa_descriptions}

**Example Writing Style:**

{example_text}

**Your task is to generate the species descriptions following the instructions above. Remember to use only the information provided without adding any new content.**
"""
    return prompt

def construct_dichotomous_key_prompt(key_features_dataset, example_dichotomous_key):
    """
    Constructs the prompt for generating the dichotomous key.
    """
    prompt = f"""You are a taxonomist tasked with creating a dichotomous key for the following species based on their key distinguishing features.

**Instructions:**

- Use only the provided key features for each species.
- Generate a dichotomous key that includes all species.
- Ensure the key is clear and follows standard formatting.

**Species Key Features:**

{key_features_dataset}

**Example Dichotomous Key:**

{example_dichotomous_key}

Please generate the dichotomous key using the information above.
"""
    return prompt

def call_gpt4_api(client, prompt, model, temperature, max_tokens):
    """
    Calls the OpenAI GPT-4 API with the provided prompt and parameters.
    Uses the updated method client.chat.completions.create.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}", file=sys.stderr)
        return None

def save_output(output_text, output_folder, output_prefix):
    """
    Saves the generated output as both .txt and .tex files.
    """
    os.makedirs(output_folder, exist_ok=True)
    txt_output_path = os.path.join(output_folder, f"{output_prefix}.txt")
    tex_output_path = os.path.join(output_folder, f"{output_prefix}.tex")

    # Save as .txt file
    try:
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Output saved to {txt_output_path}")
    except Exception as e:
        print(f"Error saving .txt file: {e}", file=sys.stderr)

    # Save as .tex file
    try:
        with open(tex_output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Output saved to {tex_output_path}")
    except Exception as e:
        print(f"Error saving .tex file: {e}", file=sys.stderr)

def estimate_token_count(text):
    """
    Estimates the number of tokens in the text.
    Simple estimation: 1 token per 4 characters.
    """
    return len(text) / 4

def extract_diagnosis_section(species_data):
    """
    Extracts the Diagnosis section from the species data.
    """
    lines = species_data.strip().split('\n')
    diagnosis_lines = []
    recording = False
    for line in lines:
        if line.strip() == 'Diagnosis:':
            recording = True
            continue
        if recording:
            if line.strip() == '':
                break  # Stop if there's an empty line
            diagnosis_lines.append(line.strip())
    return ' '.join(diagnosis_lines) if diagnosis_lines else ''

def extract_key_features(species_data):
    """
    Extracts concise key distinguishing features from the species data.
    """
    key_features = []
    # Attempt to extract from the Diagnosis section
    diagnosis = extract_diagnosis_section(species_data)
    if diagnosis:
        # Limit to the first sentence
        concise_diagnosis = diagnosis.split('. ')[0] + '.' if '. ' in diagnosis else diagnosis
        key_features.append(concise_diagnosis)
    else:
        # Fall back to QA section
        qa_section = extract_qa_section(species_data)
        if qa_section:
            # Summarize QA section (e.g., first sentence)
            concise_qa = qa_section.split('. ')[0] + '.' if '. ' in qa_section else qa_section
            key_features.append(concise_qa)
    return key_features

def create_key_features_dataset(all_species_key_features):
    """
    Creates a dataset of species and their key features.
    """
    dataset_lines = []
    for species in all_species_key_features:
        species_name = species['species_name']
        key_features = '; '.join(species['key_features'])
        line = f"{species_name}: {key_features}"
        dataset_lines.append(line)
    return '\n'.join(dataset_lines)

def read_example_dichotomous_key(file_path):
    """
    Reads the example file and extracts the dichotomous key.
    Assumes the dichotomous key starts after a line 'Dichotomous Key:'.
    """
    content = read_file(file_path)
    parts = content.split('Dichotomous Key:')
    if len(parts) > 1:
        return 'Dichotomous Key:' + parts[1]
    else:
        print("No example dichotomous key found in the example file.", file=sys.stderr)
        return ''

def main():
    args = parse_arguments()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    
    client = openai
    openai.api_key = api_key

    try:
        species_data_list = read_description_file(args.description_file)
        if not species_data_list:
            print("Description file is empty or no species data found.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error reading description file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        example_text = read_file(args.example_file)
        if not example_text:
            print("Example file is empty.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error reading example file: {e}", file=sys.stderr)
        sys.exit(1)

    model = args.model
    temperature = args.temperature
    max_tokens = args.max_tokens

    # Estimate the length of the example text
    example_text_length = estimate_token_count(example_text)

    # Define the maximum tokens per batch (input + output)
    max_tokens_per_batch = 30000  # Adjust as needed based on your species data

    # Split species data into batches for description generation
    batches = []
    current_batch = []
    current_tokens = 0

    for species_data in species_data_list:
        species_tokens = estimate_token_count(species_data)
        overhead_tokens = estimate_token_count("...") + example_text_length
        total_tokens = current_tokens + species_tokens + overhead_tokens

        if total_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [species_data]
            current_tokens = species_tokens
        else:
            current_batch.append(species_data)
            current_tokens += species_tokens

    if current_batch:
        batches.append(current_batch)

    all_outputs = []
    all_species_key_features = []

    for batch_index, batch in enumerate(batches):
        print(f"\nProcessing batch {batch_index + 1}/{len(batches)} with {len(batch)} species.")

        qa_descriptions = []
        measurements_list = []
        materials_examined_list = []
        species_names = []
        batch_species_data = []

        for species_data in batch:
            species_name = extract_species_name(species_data)
            species_names.append(species_name)

            qa_section = extract_qa_section(species_data)
            if qa_section:
                qa_descriptions.append(f"Species: {species_name}\n{qa_section}")
            else:
                qa_descriptions.append(f"Species: {species_name}\nNo QA section available.")

            measurements = extract_measurements(species_data)
            measurements_formatted = format_measurements(measurements)
            measurements_list.append(measurements_formatted)

            materials_examined = extract_materials_examined(species_data)
            materials_formatted = format_materials_examined(materials_examined)
            materials_examined_list.append(materials_formatted)

            # For key features extraction
            key_features = extract_key_features(species_data)
            all_species_key_features.append({
                'species_name': species_name,
                'key_features': key_features
            })

            batch_species_data.append(species_data)

        # Construct prompt for species descriptions
        prompt = construct_description_prompt(qa_descriptions, example_text)

        # Send request to GPT-4
        print("Sending request to GPT-4 for species descriptions...")
        descriptions_text = call_gpt4_api(client, prompt, model, temperature, max_tokens)

        if descriptions_text:
            # Split descriptions back into individual species
            descriptions = descriptions_text.strip().split('\n\n')
            combined_species_outputs = []
            for desc, meas, mat_examined in zip(descriptions, measurements_list, materials_examined_list):
                combined_output = f"{desc}\n\n{mat_examined}\n\n{meas}"
                combined_species_outputs.append(combined_output)
            all_outputs.extend(combined_species_outputs)
        else:
            print(f"Failed to get a response from GPT-4 for batch {batch_index + 1}.", file=sys.stderr)

    # After processing all batches, generate the dichotomous key for all species
    key_features_dataset = create_key_features_dataset(all_species_key_features)

    # Read the example dichotomous key from your example file
    example_dichotomous_key = read_example_dichotomous_key(args.example_file)

    # Construct the prompt for the dichotomous key
    dichotomous_key_prompt = construct_dichotomous_key_prompt(key_features_dataset, example_dichotomous_key)

    # Estimate tokens and ensure it fits within the limit
    MAX_CONTEXT_LENGTH = 30000  # Adjust based on the model you're using
    prompt_token_estimate = estimate_token_count(dichotomous_key_prompt)
    if prompt_token_estimate + args.max_tokens > MAX_CONTEXT_LENGTH:
        print("The prompt is too long. Consider further summarizing key features.", file=sys.stderr)
        sys.exit(1)

    # Send the request to GPT-4 for the dichotomous key
    print("Generating dichotomous key including all species...")
    dichotomous_key = call_gpt4_api(client, dichotomous_key_prompt, model, temperature, args.max_tokens)

    # Combine outputs
    final_output_text = "\n\n".join(all_outputs)
    if dichotomous_key:
        final_output_text += "\n\nDichotomous Key:\n" + dichotomous_key
    else:
        print("Failed to generate the dichotomous key.", file=sys.stderr)

    if final_output_text:
        # Save the final combined output
        save_output(final_output_text, args.output_folder, args.output_prefix)
    else:
        print("Failed to generate the final output.", file=sys.stderr)

if __name__ == "__main__":
    main()
