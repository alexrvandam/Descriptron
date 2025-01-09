import argparse
import openai
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI's model with a JSONL file.")
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the JSONL file for fine-tuning.')
    parser.add_argument('--model', type=str, required=True, help='Base model to fine-tune (e.g., gpt-4o-2024-08-06).')
    args = parser.parse_args()

    # Ensure the JSONL file exists
    if not os.path.isfile(args.jsonl_file):
        print(f"Error: The file {args.jsonl_file} does not exist.")
        return

    # Initialize the OpenAI client
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    try:
        # Upload the JSONL file
        with open(args.jsonl_file, "rb") as file:
            upload_response = client.files.create(
                file=file,
                purpose='fine-tune'
            )
        
        # Retrieve the file ID directly from the response
        file_id = upload_response.id
        print(f"File uploaded successfully. File ID: {file_id}")

        # Create a fine-tuning job
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=args.model
        )

        # Retrieve the job ID directly from the response
        fine_tune_job_id = fine_tune_response.id
        print(f"Fine-tuning job created successfully. Job ID: {fine_tune_job_id}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
