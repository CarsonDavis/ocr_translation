#!/usr/bin/env python3
"""
clean.py - Script to clean OCR-generated markdown using AI models
"""

import os
import argparse
import aisuite as ai
from pathlib import Path


# Default prompts for cleaning
CLEANING_SYSTEM_PROMPT = "You are a deliberate and careful editor of old French"
CLEANING_USER_PROMPT = (
    "I created the following French 1500s text with OCR, and it might have missed "
    "some characters or made minor mistakes. Correct anything you see wrong, and "
    "respond with only the corrected information. Maintain the markdown formatting "
    "of the original."
)

# Available models
MODELS = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]


def call_llm(model, system_prompt, user_prompt, text, temperature=0.75):
    """
    Call an LLM using the aisuite client.

    Args:
        model (str): The LLM model to use
        system_prompt (str): The system prompt to use
        user_prompt (str): The user prompt to use
        text (str): The text to send to the LLM
        temperature (float, optional): The temperature to use. Defaults to 0.75.

    Returns:
        str: The LLM's response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\n{text}"},
    ]

    client = ai.Client()
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )

    return response.choices[0].message.content


def read_markdown(file_path):
    """
    Read a markdown file.

    Args:
        file_path (str): Path to the markdown file

    Returns:
        str or None: Contents of the file or None if reading fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading markdown file: {e}")
        return None


def save_markdown(markdown_text, output_path):
    """
    Save markdown text to a file.

    Args:
        markdown_text (str): Markdown text to save
        output_path (str): Path to save the markdown file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write markdown to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        print(f"Saved cleaned markdown to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving markdown: {e}")
        return False


def clean_markdown_with_llm(
    input_path,
    output_path="cleaned",
    model="openai:gpt-4o",
    system_prompt=CLEANING_SYSTEM_PROMPT,
    user_prompt=CLEANING_USER_PROMPT,
    temperature=0.75,
):
    """
    Clean a markdown file using an LLM.

    Args:
        input_path (str): Path to the input markdown file
        output_path (str, optional): Path to save the cleaned markdown file.
                                    If None, appends '_cleaned' to the input path.
        model (str, optional): LLM model to use. Defaults to claude-3-5-sonnet.
        system_prompt (str, optional): Custom system prompt. Defaults to CLEANING_SYSTEM_PROMPT.
        user_prompt (str, optional): Custom user prompt. Defaults to CLEANING_USER_PROMPT.
        temperature (float, optional): Temperature for LLM generation. Defaults to 0.75.

    Returns:
        dict: Results dictionary with paths and success status
    """
    results = {
        "success": False,
        "input_path": input_path,
        "output_path": output_path,
        "cleaned_text": None,
    }

    # Read the markdown file
    markdown_text = read_markdown(input_path)
    if markdown_text is None:
        return results

    # Generate output path if it's a directory
    input_path_obj = Path(input_path)
    if os.path.isdir(output_path) or output_path == "cleaned":
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Put the file in the output directory with the same name
        output_path = str(Path(output_path) / input_path_obj.name)

    results["output_path"] = output_path

    try:
        # Call the LLM to clean the text
        print(f"Cleaning markdown using {model}...")
        cleaned_text = call_llm(
            model, system_prompt, user_prompt, markdown_text, temperature
        )
        results["cleaned_text"] = cleaned_text

        # Save the cleaned text
        if save_markdown(cleaned_text, output_path):
            results["success"] = True

    except Exception as e:
        print(f"Error during cleaning process: {e}")

    return results


def batch_clean_directory(
    input_dir,
    output_dir="cleaned",
    file_pattern="*.md",
    model="openai:gpt-4o",
    system_prompt=CLEANING_SYSTEM_PROMPT,
    user_prompt=CLEANING_USER_PROMPT,
    temperature=0.75,
):
    """
    Clean all markdown files in a directory.

    Args:
        input_dir (str): Directory containing markdown files to clean
        output_dir (str, optional): Directory to save cleaned files.
                                   If None, files are saved alongside originals with '_cleaned' suffix.
        file_pattern (str, optional): Pattern to match files. Defaults to "*.md".
        model (str, optional): LLM model to use. Defaults to claude-3-5-sonnet.
        system_prompt (str, optional): Custom system prompt. Defaults to CLEANING_SYSTEM_PROMPT.
        user_prompt (str, optional): Custom user prompt. Defaults to CLEANING_USER_PROMPT.
        temperature (float, optional): Temperature for LLM generation. Defaults to 0.75.

    Returns:
        list: List of results dictionaries for each file
    """
    results = []

    # Get all markdown files in the input directory
    input_path = Path(input_dir)
    markdown_files = list(input_path.glob(file_pattern))

    if not markdown_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(markdown_files)} markdown files to clean")

    # Process each file
    for file_path in markdown_files:
        # Create output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Construct output path in the output directory
            rel_path = file_path.relative_to(input_path)
            output_path = Path(output_dir) / rel_path
            output_path = str(output_path)
        else:
            # This should not happen with the new default
            output_path = None

        # Clean the file
        file_result = clean_markdown_with_llm(
            str(file_path), output_path, model, system_prompt, user_prompt, temperature
        )

        results.append(file_result)

    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"Cleaning complete. {successful}/{len(results)} files successfully cleaned.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean OCR-generated markdown files using AI"
    )

    # Input and output options
    parser.add_argument("input", help="Input markdown file or directory")
    parser.add_argument(
        "--output",
        "-o",
        default="cleaned",
        help="Output markdown file or directory (default: 'cleaned' directory)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all markdown files in input directory",
    )
    parser.add_argument(
        "--pattern",
        default="*.md",
        help="File pattern when using batch mode (default: *.md)",
    )

    # Model and prompt options
    parser.add_argument(
        "--model",
        "-m",
        default="openai:gpt-4o",
        choices=MODELS,
        help="AI model to use for cleaning",
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        default=CLEANING_SYSTEM_PROMPT,
        help="Custom system prompt for cleaning",
    )
    parser.add_argument(
        "--user-prompt",
        "-u",
        default=CLEANING_USER_PROMPT,
        help="Custom user prompt for cleaning",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.75,
        help="Temperature for LLM generation (0.0-1.0)",
    )

    args = parser.parse_args()

    # Determine if we're processing a single file or a directory
    if args.batch or os.path.isdir(args.input):
        batch_clean_directory(
            args.input,
            args.output,
            args.pattern,
            args.model,
            args.system_prompt,
            args.user_prompt,
            args.temperature,
        )
    else:
        clean_markdown_with_llm(
            args.input,
            args.output,
            args.model,
            args.system_prompt,
            args.user_prompt,
            args.temperature,
        )
