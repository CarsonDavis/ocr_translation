#!/usr/bin/env python3
"""
translate.py - Script to translate markdown from French to English using AI models
"""

import os
import argparse
from pathlib import Path

# Import from utility module
import utils


# Default prompts for translation
TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional translator specializing in Old French to modern English"
)
TRANSLATION_USER_PROMPT = (
    "Translate the following Old French text to modern English. Maintain the markdown "
    "formatting of the original. Ensure the translation captures both the meaning and "
    "the style of the original text."
)

# Available models
MODELS = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]


def translate_markdown(
    input_path,
    output_path=None,
    model="openai:gpt-4o",
    system_prompt=TRANSLATION_SYSTEM_PROMPT,
    user_prompt=TRANSLATION_USER_PROMPT,
    temperature=0.75,
):
    """
    Translate a markdown file from French to English using an LLM.

    Args:
        input_path (str): Path to the input markdown file
        output_path (str, optional): Path to save the translated markdown file.
                                     If None, creates an appropriate path.
        model (str, optional): LLM model to use. Defaults to "openai:gpt-4o".
        system_prompt (str, optional): Custom system prompt. Defaults to TRANSLATION_SYSTEM_PROMPT.
        user_prompt (str, optional): Custom user prompt. Defaults to TRANSLATION_USER_PROMPT.
        temperature (float, optional): Temperature for LLM generation. Defaults to 0.75.

    Returns:
        dict: Results dictionary with paths and success status
    """
    results = {
        "success": False,
        "input_path": input_path,
        "output_path": output_path,
        "translated_text": None,
    }

    # Read the markdown file
    markdown_text = utils.read_markdown(input_path)
    if markdown_text is None:
        return results

    # Generate appropriate output path if none provided
    if output_path is None:
        output_dir = "translated"
        output_path = utils.get_output_path(input_path, output_dir)

    results["output_path"] = output_path

    try:
        # Create AI client
        ai_client = utils.TextAI(model=model)

        # Call the LLM to translate the text
        print(f"Translating markdown using {model}...")
        translated_text = ai_client.call(
            system_prompt, user_prompt, markdown_text, temperature
        )
        results["translated_text"] = translated_text

        # Save the translated text
        if utils.save_markdown(translated_text, output_path):
            results["success"] = True

    except Exception as e:
        print(f"Error during translation process: {e}")

    return results


def batch_translate_directory(
    input_dir,
    output_dir="translated",
    file_pattern="*.md",
    model="openai:gpt-4o",
    system_prompt=TRANSLATION_SYSTEM_PROMPT,
    user_prompt=TRANSLATION_USER_PROMPT,
    temperature=0.75,
):
    """
    Translate all markdown files in a directory from French to English.

    Args:
        input_dir (str): Directory containing markdown files to translate
        output_dir (str, optional): Directory to save translated files.
        file_pattern (str, optional): Pattern to match files. Defaults to "*.md".
        model (str, optional): LLM model to use. Defaults to openai:gpt-4o.
        system_prompt (str, optional): Custom system prompt. Defaults to TRANSLATION_SYSTEM_PROMPT.
        user_prompt (str, optional): Custom user prompt. Defaults to TRANSLATION_USER_PROMPT.
        temperature (float, optional): Temperature for LLM generation. Defaults to 0.75.

    Returns:
        list: List of results dictionaries for each file
    """
    results = []

    # Get all markdown files in the input directory
    markdown_files = utils.find_files(input_dir, file_pattern)

    if not markdown_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(markdown_files)} markdown files to translate")

    # Create output directory if it doesn't exist
    utils.ensure_dir(output_dir)

    # Process each file
    for file_path in markdown_files:
        # Construct output path in the output directory
        rel_path = (
            file_path.relative_to(Path(input_dir))
            if Path(input_dir) in file_path.parents
            else file_path.name
        )
        output_path = Path(output_dir) / rel_path

        # Translate the file
        print(f"Translating {file_path}...")
        file_result = translate_markdown(
            str(file_path),
            str(output_path),
            model,
            system_prompt,
            user_prompt,
            temperature,
        )

        results.append(file_result)

    # Print summary
    utils.print_batch_summary(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate markdown files from French to English using AI"
    )

    # Input and output options
    parser.add_argument("input", help="Input markdown file or directory")
    parser.add_argument(
        "--output",
        "-o",
        default="translated",
        help="Output markdown file or directory (default: 'translated' directory)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all markdown files in input directory",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*.md",
        help="File pattern when using batch mode (default: *.md)",
    )

    # Model and prompt options
    parser.add_argument(
        "--model",
        "-m",
        default="openai:gpt-4o",
        choices=MODELS,
        help="AI model to use for translation",
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        default=TRANSLATION_SYSTEM_PROMPT,
        help="Custom system prompt for translation",
    )
    parser.add_argument(
        "--user-prompt",
        "-u",
        default=TRANSLATION_USER_PROMPT,
        help="Custom user prompt for translation",
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
        batch_translate_directory(
            args.input,
            args.output,
            args.pattern,
            args.model,
            args.system_prompt,
            args.user_prompt,
            args.temperature,
        )
    else:
        result = translate_markdown(
            args.input,
            args.output,
            args.model,
            args.system_prompt,
            args.user_prompt,
            args.temperature,
        )

        if result["success"]:
            print(f"Successfully translated {args.input}")
            print(f"Translated markdown saved to {result['output_path']}")
        else:
            print(f"Failed to translate {args.input}")
