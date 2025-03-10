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
from utils.constants import (
    DEFAULT_TRANSLATED_DIR,
    DEFAULT_MD_PATTERN,
    DEFAULT_TRANSLATE_MODEL,
    DEFAULT_TEMPERATURE,
    AVAILABLE_MODELS,
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_PROMPT,
)


def translate_markdown(
    input_path: str,
    output_path: str | None = None,
    model: str = DEFAULT_TRANSLATE_MODEL,
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
                     If a directory name/path, file will be created within that directory.
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
        "output_path": None,
        "translated_text": None,
    }

    # Read the markdown file
    markdown_text = file_handling.read_markdown(input_path)
    if markdown_text is None:
        return results

    # Always use get_output_path to ensure consistent behavior
    # whether output_path is a directory or file path
    output_path = file_handling.get_output_path(
        input_path, output_path or DEFAULT_TRANSLATED_DIR
    )
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
    output_dir: str = DEFAULT_TRANSLATED_DIR,
    file_pattern: str = DEFAULT_MD_PATTERN,
    model: str = DEFAULT_TRANSLATE_MODEL,
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
        default=DEFAULT_TRANSLATED_DIR,
        help=f"Output directory (default: '{DEFAULT_TRANSLATED_DIR}'). "
        f"Files will be saved within this directory with their original names.",
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
        default=DEFAULT_MD_PATTERN,
        help=f"File pattern when using batch mode (default: '{DEFAULT_MD_PATTERN}')",
    )

    # Model and prompt options
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_TRANSLATE_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"AI model to use for translation (default: '{DEFAULT_TRANSLATE_MODEL}')",
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
        # For single file, output path will be handled by get_output_path
        # which will create a file within the output directory
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
