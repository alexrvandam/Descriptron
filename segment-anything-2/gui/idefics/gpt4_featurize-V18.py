import argparse
import os
import openai
import base64
from tqdm import tqdm
import json
import sys
from glob import glob
from datetime import datetime
import csv
from pathlib import Path
import time
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automate species description generation using GPT-4 with VQA.")
    parser.add_argument('--image_path', type=str, help='Path to a single image with overlays.')
    parser.add_argument('--image_folder', type=str, help='Path to a folder containing images with overlays.')
    parser.add_argument('--questions_file', type=str, required=True, help='Path to the text file containing questions.')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the Q&A results (JSON format) for each image.')

    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature to use (default: 0.2).')
    parser.add_argument('--max_tokens', type=int, default=150, help='Maximum number of tokens to generate in the answer (default: 150).')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use for inference (default: gpt-4o).')

    # Seed/preface options
    parser.add_argument('--preface', type=str, default="",
                        help="Optional text to prepend to each question. If none provided, a default is used.")
    parser.add_argument('--preface_file', type=str, default="",
                        help="Optional path to a text file with the preface/seed prompt.")

    # Sweep options
    parser.add_argument('--sweep_temperatures', type=str, default="",
                        help='Comma-separated temperatures to sweep, e.g. "0.0,0.2,0.5". If empty, uses --temperature only.')
    parser.add_argument('--replicates', type=int, default=1,
                        help='Number of replicates per (image, temperature). Default 1 (backward compatible).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional seed passed to the API (if supported). If omitted, sampling will vary naturally.')

    # NEW: skip existing
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip images that already have an output JSON in the target temperature/replicate folder.')

    return parser.parse_args()

def load_questions(questions_file):
    if not os.path.isfile(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    with open(questions_file, 'r', encoding='utf-8') as f:
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

def with_retries(fn, tries=3, base_delay=1.0, jitter=0.5):
    """Exponential backoff retries for transient API errors."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            print(f"Retry {attempt}/{tries} after error: {e} (sleep {delay:.1f}s)", file=sys.stderr)
            time.sleep(delay)

def ask_gpt4(client, model, base64_image, question, temperature, max_tokens, user_preface, seed=None):
    # Normalize and fall back to a default seed/preface if empty
    user_preface = (user_preface or "").strip() or (
        "INSTRUCTIONS:\n"
        "- Colored contours and matching text labels are overlays that mark regions of interest (ROIs) for specific morphological features.\n"
        "- The contour/label is NOT part of the specimen. Do not treat its color or stroke as pigmentation or structure.\n"
        "- Base all observations ONLY on the pixels inside the contour. You may describe the outer shape by following the contour boundary.\n"
        "- Ignore overlay artifacts (contours, labels, arrows, legends, watermarks, scale graphics) except to locate the ROI.\n"
        "- If the requested detail is not visible or uncertain, say so explicitly."
    )
    combined_text = f"{user_preface}\n\n{question}"

    try:
        kwargs = dict(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_text},
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
        if seed is not None:
            kwargs["seed"] = seed

        def _call():
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()

        return with_retries(_call, tries=3, base_delay=1.0)
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}", file=sys.stderr)
        return "Error: Unable to retrieve answer."

def process_image(image_path, questions, client, model, out_dir, temperature, max_tokens, user_preface,
                  seed=None, manifest_writer=None, rep_idx=1, skip_existing=False):
    # Compute output path once and reuse it in manifest rows
    out_json = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_description.json")

    # NEW: skip existing outputs
    if skip_existing and os.path.exists(out_json) and os.path.getsize(out_json) > 0:
        print(f"Skipping existing output: {out_json}")
        return

    try:
        base64_image = encode_image_to_base64(image_path)
        print(f"Image loaded and encoded successfully from {image_path}.")
    except Exception as e:
        print(f"Error encoding image: {e}", file=sys.stderr)
        return

    qa_results = {}
    for idx, question in enumerate(questions, 1):
        print(f"\nProcessing question {idx}: {question}")
        answer = ask_gpt4(client, model, base64_image, question, temperature, max_tokens, user_preface, seed=seed)
        qa_results[f"Q{idx}: {question}"] = answer
        print(f"Answer: {answer}")

        if manifest_writer is not None:
            manifest_writer.writerow({
                "image": os.path.abspath(image_path),
                "model": model,
                "temperature": temperature,
                "replicate": rep_idx,
                "question_index": idx,
                "question": question,
                "answer": answer,
                "output_json": os.path.abspath(out_json)
            })

    try:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, indent=4, ensure_ascii=False)
        print(f"\nQ&A results saved to {out_json}")
    except Exception as e:
        print(f"Error saving results to file: {e}", file=sys.stderr)
        return

    # Optional provenance sidecar
    if manifest_writer is not None:
        with open(out_json + ".meta.json", "w", encoding='utf-8') as mf:
            json.dump({
                "image": os.path.abspath(image_path),
                "model": model,
                "temperature": temperature,
                "replicate": rep_idx,
                "output_json": os.path.abspath(out_json)
            }, mf, indent=2, ensure_ascii=False)

def iter_images(image_path, image_folder):
    if image_path:
        yield image_path
    else:
        # QoL: compact one-liner (png/jpg/jpeg, both cases)
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        files = [p for pat in patterns for p in glob(os.path.join(image_folder, pat))]
        return files

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

    model = args.model
    max_tokens = args.max_tokens

    # Load preface from file if provided (prefer inline --preface if both exist)
    user_preface = args.preface or ""
    if args.preface_file:
        try:
            with open(args.preface_file, 'r', encoding='utf-8') as fh:
                file_text = fh.read()
            if not user_preface.strip():
                user_preface = file_text
        except Exception as e:
            print(f"Warning: could not read --preface_file: {e}", file=sys.stderr)

    # Build temperature list
    if args.sweep_temperatures.strip():
        temps = [float(t.strip()) for t in args.sweep_temperatures.split(",") if t.strip()]
    else:
        temps = [float(args.temperature)]
    replicates = max(1, int(args.replicates))

    # Provenance / metadata
    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "max_tokens": max_tokens,
        "preface_used": bool((user_preface or "").strip()),
        "temperatures": temps,
        "replicates": replicates,
        "seed": args.seed,
        "skip_existing": bool(args.skip_existing),
        "source_images": "file" if args.image_path else "folder",
        "image_path": args.image_path,
        "image_folder": args.image_folder,
        "questions_file": os.path.abspath(args.questions_file)
    }
    with open(os.path.join(args.output_folder, "run_meta.json"), "w", encoding='utf-8') as mf:
        json.dump(run_meta, mf, indent=2, ensure_ascii=False)

    # Manifest for ROUGE / analysis
    manifest_path = os.path.join(args.output_folder, "manifest.csv")
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=[
        "image","model","temperature","replicate","question_index","question","answer","output_json"
    ])
    manifest_writer.writeheader()

    # Collect image list
    if args.image_path:
        images = [args.image_path]
    else:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        images = [p for pat in patterns for p in glob(os.path.join(args.image_folder, pat))]
        if not images:
            print(f"No images found in the folder {args.image_folder}", file=sys.stderr)
            manifest_file.close()
            sys.exit(1)

    # Main sweep
    for T in temps:
        for rep in range(1, replicates + 1):
            # Organize outputs as OUTPUT_DIR/T_x.xx/rep_k/
            T_dir = os.path.join(args.output_folder, f"T_{T:.2f}", f"rep_{rep}")
            Path(T_dir).mkdir(parents=True, exist_ok=True)
            print(f"\n=== Running temperature {T:.2f}, replicate {rep}/{replicates} ===")

            # If you want non-deterministic replicates even with seed, vary seed per replicate:
            effective_seed = args.seed if args.seed is None else (args.seed + rep - 1)

            for image_path in tqdm(images, desc=f"T={T:.2f}, rep={rep}"):
                process_image(
                    image_path=image_path,
                    questions=questions,
                    client=client,
                    model=model,
                    out_dir=T_dir,
                    temperature=T,
                    max_tokens=max_tokens,
                    user_preface=user_preface,
                    seed=effective_seed,
                    manifest_writer=manifest_writer,
                    rep_idx=rep,
                    skip_existing=args.skip_existing
                )

    manifest_file.close()
    print(f"\nManifest written to: {manifest_path}")
    print("Done.")

if __name__ == "__main__":
    main()

