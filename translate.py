#!/usr/bin/env python3
"""
translate.py - Script to translate markdown from French to English using AI models
"""

import os
import argparse
from pathlib import Path
from typing import Any

# Import from utility module
import utils.utils as utils
from utils import file_handling


# Default values as constants
DEFAULT_OUTPUT_DIR = "translated"
DEFAULT_FILE_PATTERN = "*.md"
DEFAULT_MODEL = "openai:gpt-4o"
DEFAULT_TEMPERATURE = 0.75

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
    input_path: str,
    output_path: str | None = None,
    model: str = DEFAULT_MODEL,
    system_prompt: str = TRANSLATION_SYSTEM_PROMPT,
    user_prompt: str = TRANSLATION_USER_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, bool | str | None]:
    """
    Translate a markdown file from French to English using an LLM.

    Args:
        input_path: Path to the input markdown file
        output_path: Path to save the translated markdown file.
                     If None, creates an appropriate path.
        model: LLM model to use
        system_prompt: Custom system prompt
        user_prompt: Custom user prompt
        temperature: Temperature for LLM generation

    Returns:
        dict: Results dictionary with paths and success status
    """
    results: dict[str, bool | str | None] = {
        "success": False,
        "input_path": input_path,
        "output_path": output_path,
        "translated_text": None,
    }

    # Read the markdown file
    markdown_text = file_handling.read_markdown(input_path)
    if markdown_text is None:
        return results

    # Generate appropriate output path if none provided
    if output_path is None:
        output_path = file_handling.get_output_path(input_path, DEFAULT_OUTPUT_DIR)

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
        if file_handling.save_markdown(translated_text, output_path):
            results["success"] = True

    except Exception as e:
        print(f"Error during translation process: {e}")

    return results


def batch_translate_directory(
    input_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    model: str = DEFAULT_MODEL,
    system_prompt: str = TRANSLATION_SYSTEM_PROMPT,
    user_prompt: str = TRANSLATION_USER_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict[str, bool | str | None]]:
    """
    Translate all markdown files in a directory from French to English.

    Args:
        input_dir: Directory containing markdown files to translate
        output_dir: Directory to save translated files
        file_pattern: Pattern to match files
        model: LLM model to use
        system_prompt: Custom system prompt
        user_prompt: Custom user prompt
        temperature: Temperature for LLM generation

    Returns:
        list: List of results dictionaries for each file
    """
    results: list[dict[str, bool | str | None]] = []

    # Get all markdown files in the input directory
    markdown_files = utils.find_files(input_dir, file_pattern)

    if not markdown_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(markdown_files)} markdown files to translate")

    # Create output directory if it doesn't exist
    file_handling.ensure_dir(output_dir)

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
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output markdown file or directory (default: '{DEFAULT_OUTPUT_DIR}')",
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
        default=DEFAULT_FILE_PATTERN,
        help=f"File pattern when using batch mode (default: '{DEFAULT_FILE_PATTERN}')",
    )

    # Model and prompt options
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        choices=MODELS,
        help=f"AI model to use for translation (default: '{DEFAULT_MODEL}')",
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
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for LLM generation (0.0-1.0) (default: {DEFAULT_TEMPERATURE})",
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
