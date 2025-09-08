import argparse
import openai
import os
import json
import tempfile

def _inject_seed_into_jsonl(in_path, seed_text, out_path=None, force=False):
    """
    Reads chat-format JSONL and ensures each line has a system message
    at the beginning. Returns the rewritten file path and a small summary.
    Lines that don't have 'messages' are left as-is (and counted).
    """
    total = 0
    modified = 0
    out_fh = None
    created_tmp = False

    if out_path is None:
        # write next to the source so it's easy to inspect
        d, b = os.path.split(in_path)
        base, ext = os.path.splitext(b)
        out_path = os.path.join(d, f"{base}.with_seed{ext}")
        created_tmp = True

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        out_fh = fout
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            msgs = obj.get("messages")
            if isinstance(msgs, list):
                has_system = any((m.get("role") == "system") for m in msgs)
                if not has_system or force:
                    obj["messages"] = [{"role": "system", "content": seed_text}] + msgs
                    modified += 1
            # If it's a completions-style record (prompt/completion), we can’t safely
            # inject a chat 'system' without converting formats, so we leave it alone.

            out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return out_path, {"total": total, "modified": modified, "left_unmodified": total - modified, "created_tmp": created_tmp}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI's model with a JSONL file.")
    parser.add_argument('--jsonl_file', type=str, required=True, help='Path to the JSONL file for fine-tuning.')
    parser.add_argument('--model', type=str, required=True, help='Base model to fine-tune (e.g., gpt-4o-2024-08-06).')

    # NEW: seed prompt options
    parser.add_argument('--seed_prompt_file', type=str, default=None,
                        help='Optional path to a text file whose contents will be injected as a system message into each training example (chat-format).')
    parser.add_argument('--seed_prompt', type=str, default=None,
                        help='Optional literal seed prompt text. If provided, overrides --seed_prompt_file.')
    parser.add_argument('--force_seed', action='store_true',
                        help='Prepend the system message even if an example already has one.')

    # NEW: optional path for the rewritten JSONL
    parser.add_argument('--out_jsonl', type=str, default=None,
                        help='Where to save the rewritten JSONL (default: alongside the input with ".with_seed.jsonl").')

    args = parser.parse_args()

    # Ensure the JSONL file exists
    if not os.path.isfile(args.jsonl_file):
        print(f"Error: The file {args.jsonl_file} does not exist.")
        return

    # Initialize the OpenAI client
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI()

    # NEW: prepare training file with seed prompt if requested
    training_path = args.jsonl_file
    if args.seed_prompt or args.seed_prompt_file:
        seed_text = args.seed_prompt
        if not seed_text and args.seed_prompt_file:
            if not os.path.isfile(args.seed_prompt_file):
                print(f"Error: seed_prompt_file '{args.seed_prompt_file}' does not exist.")
                return
            with open(args.seed_prompt_file, "r", encoding="utf-8") as fh:
                seed_text = fh.read().strip()
        if not seed_text:
            print("Error: Seed prompt is empty.")
            return

        rewritten_path, info = _inject_seed_into_jsonl(
            training_path,
            seed_text,
            out_path=args.out_jsonl,
            force=args.force_seed
        )
        print(f"[seed] Injected system prompt into {info['modified']}/{info['total']} lines "
              f"(left unmodified: {info['left_unmodified']}).")
        print(f"[seed] Using rewritten training file: {rewritten_path}")
        training_path = rewritten_path

    try:
        # Upload the (possibly rewritten) JSONL file
        with open(training_path, "rb") as file:
            upload_response = client.files.create(
                file=file,
                purpose='fine-tune'
            )

        file_id = upload_response.id
        print(f"File uploaded successfully. File ID: {file_id}")

        # Create a fine-tuning job
        fine_tune_response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=args.model
        )
        fine_tune_job_id = fine_tune_response.id
        print(f"Fine-tuning job created successfully. Job ID: {fine_tune_job_id}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
