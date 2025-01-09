import argparse
import os
import openai
import sys
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate species description, diagnosis, and dichotomous key using GPT-4.")
    parser.add_argument('--description_file', type=str, required=True, help='Path to the description text file.')
    parser.add_argument('--example_file', type=str, required=True, help='Path to the example text file serving as a writing style template.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the generated outputs (both .txt and .tex formats).')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for the output files (default: output).')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature to use (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=2000, help='Maximum number of tokens to generate in the output (default: 2000).')
    parser.add_argument('--model', type=str, default='gpt-4', help='Model to use for inference (default: gpt-4).')
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

def construct_prompt(species_data, example_text):
    prompt = f"""You are provided with species data for multiple species and an example of the desired writing style.

**Instructions:**

1. **Do not fabricate or make up any information.** Use only the data provided in the species data.

2. **Rearrange the ordering** of the content if necessary to improve readability, but do not add any new information.

3. **Generate a comprehensive Dichotomous Key** down to the species level using nested species groups based on forewing characters and other morphological features, considering all species provided.

4. **Note:** If "male" or "female" is in the species name and they have a similar name, they are male and female of the same species.

5. **Follow the style and format** of the provided example.

**Species Data:**

{species_data}

**Example Writing Style:**

{example_text}

**Your task is to generate the output following the instructions above. Remember to use only the information provided without adding any new content.**
"""
    return prompt

def call_gpt4_api(client, prompt, model, temperature, max_tokens):
    """
    Calls the OpenAI GPT-4 API with the provided prompt and parameters.
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
    # Simple estimation: 1 token per 4 characters
    return len(text) / 4

def split_species_data_into_batches(species_data_list, max_tokens_per_batch, example_text_length):
    batches = []
    current_batch = []
    current_tokens = 0

    for species_data in species_data_list:
        species_tokens = estimate_token_count(species_data)
        # Add tokens for the prompt structure and example text
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

    return batches

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
    max_tokens_per_batch = 15000  # Adjust as needed

    # Split species data into batches
    batches = split_species_data_into_batches(species_data_list, max_tokens_per_batch, example_text_length)

    all_outputs = []

    for batch_index, batch in enumerate(batches):
        print(f"\nProcessing batch {batch_index + 1}/{len(batches)} with {len(batch)} species.")
        combined_species_data = "\n\n".join(batch)

        # Save the combined species data for this batch
        combined_data_path = os.path.join(args.output_folder, f"combined_species_data_batch_{batch_index + 1}.txt")
        try:
            with open(combined_data_path, 'w', encoding='utf-8') as f:
                f.write(combined_species_data)
            print(f"Combined species data for batch {batch_index + 1} saved to {combined_data_path}")
        except Exception as e:
            print(f"Error saving combined species data for batch {batch_index + 1}: {e}", file=sys.stderr)

        # Construct prompt with combined species data
        prompt = construct_prompt(combined_species_data, example_text)

        print("Sending request to GPT-4...")
        output_text = call_gpt4_api(client, prompt, model, temperature, max_tokens)

        if output_text:
            all_outputs.append(output_text)
        else:
            print(f"Failed to get a response from GPT-4 for batch {batch_index + 1}.", file=sys.stderr)

    # Combine outputs from all batches
    final_output_text = "\n\n".join(all_outputs)

    if final_output_text:
        # Save the final combined output
        save_output(final_output_text, args.output_folder, args.output_prefix)
    else:
        print("Failed to get any responses from GPT-4.", file=sys.stderr)

if __name__ == "__main__":
    main()
