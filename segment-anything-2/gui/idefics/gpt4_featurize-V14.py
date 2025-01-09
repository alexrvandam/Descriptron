import argparse
import os
import openai
import base64
from tqdm import tqdm
import json
import sys
from glob import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automate species description generation using GPT-4 with VQA.")
    parser.add_argument('--image_path', type=str, help='Path to a single image with overlays.')
    parser.add_argument('--image_folder', type=str, help='Path to a folder containing images with overlays.')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to the text file containing questions.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the Q&A results (JSON format) for each image.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature to use (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum number of tokens to generate in the answer (default: 150).')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for inference (default: gpt-4o).')
    return parser.parse_args()

def load_questions(questions_file):
    if not os.path.isfile(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    with open(questions_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def encode_image_to_base64(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        raise ValueError(f"Failed to encode image: {e}")

def ask_gpt4(client, model, base64_image, question, temperature, max_tokens):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}", file=sys.stderr)
        return "Error: Unable to retrieve answer."

def process_image(image_path, questions, client, model, output_folder, temperature, max_tokens):
    try:
        base64_image = encode_image_to_base64(image_path)
        print(f"Image loaded and encoded successfully from {image_path}.")
    except Exception as e:
        print(f"Error encoding image: {e}", file=sys.stderr)
        return

    qa_results = {}
    for idx, question in enumerate(questions, 1):
        print(f"\nProcessing question {idx}: {question}")
        
        answer = ask_gpt4(client, model, base64_image, question, temperature, max_tokens)
        qa_results[f"Q{idx}: {question}"] = answer
        print(f"Answer: {answer}")

    output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_description.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(qa_results, f, indent=4)
        print(f"\nQ&A results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to file: {e}", file=sys.stderr)

def main():
    args = parse_arguments()

    if not args.image_path and not args.image_folder:
        print("Error: You must specify either --image_path for a single image or --image_folder for a folder of images.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    openai.api_key = api_key
    client = openai

    try:
        questions = load_questions(args.questions_file)
        if not questions:
            print("No questions found in the questions file.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error loading questions: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)

    model = args.model  # Use your default model
    temperature = args.temperature
    max_tokens = args.max_tokens

    if args.image_path:
        print(f"Processing single image: {args.image_path}")
        process_image(args.image_path, questions, client, model, args.output_folder, temperature, max_tokens)
    
    elif args.image_folder:
        image_files = glob(os.path.join(args.image_folder, "*.[pj][pn]g"))
        if not image_files:
            print(f"No images found in the folder {args.image_folder}", file=sys.stderr)
            sys.exit(1)
        
        for image_path in tqdm(image_files, desc="Processing images"):
            print(f"\nProcessing image: {image_path}")
            process_image(image_path, questions, client, model, args.output_folder, temperature, max_tokens)

if __name__ == "__main__":
    main()
