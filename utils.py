#!/usr/bin/env python3
"""
utils.py - Shared utility functions for OCR, cleaning, and translation
"""

import os
import base64
from pathlib import Path


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

        print(f"Saved markdown to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving markdown: {e}")
        return False


def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory (str): Directory path to ensure exists

    Returns:
        Path: Path object of the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_output_path(input_path, output_dir, suffix=None):
    """
    Generate an appropriate output path based on input path and output directory.

    Args:
        input_path (str): Path to the input file
        output_dir (str): Directory for output
        suffix (str, optional): Suffix to add to filename. Defaults to None.

    Returns:
        str: Generated output path
    """
    input_path_obj = Path(input_path)

    # Create output directory if it doesn't exist
    ensure_dir(output_dir)

    # Generate base filename, with optional suffix
    base_name = input_path_obj.stem
    if suffix:
        base_name = f"{base_name}_{suffix}"

    # Create full output path
    return str(Path(output_dir) / f"{base_name}{input_path_obj.suffix}")


def find_files(directory, pattern="*.md"):
    """
    Find files matching a pattern in a directory.

    Args:
        directory (str): Directory to search
        pattern (str, optional): Glob pattern. Defaults to "*.md".

    Returns:
        list: List of Path objects for matching files
    """
    return list(Path(directory).glob(pattern))


def print_batch_summary(results):
    """
    Print a summary of batch processing results.

    Args:
        results (list): List of result dictionaries, each with a 'success' key
    """
    successful = sum(1 for r in results if r.get("success", False))
    print(
        f"Processing complete. {successful}/{len(results)} files successfully processed."
    )


class TextAI:
    """Client for text-based AI services (OpenAI, Anthropic, etc.)"""

    def __init__(self, api_key=None, model="openai:gpt-4o"):
        # Store the model, but ignore api_key as aisuite uses environment variables
        self.model = model

        # Import aisuite here to avoid loading it unnecessarily
        import aisuite as ai

        # Create client without passing api_key - it will use environment variables
        self.client = ai.Client()

    def call(self, system_prompt, user_prompt, text, temperature=0.75):
        """
        Call an LLM using the aisuite client.

        Args:
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

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling AI service: {e}")
            return None


class OCRAI:
    """Client for OCR-based AI services (Mistral OCR)"""

    def __init__(self, api_key=None, model="mistral-ocr-latest"):
        self.model = model

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided and MISTRAL_API_KEY environment variable not set"
                )

        # Import Mistral client here to avoid loading it unnecessarily
        from mistralai import Mistral

        self.client = Mistral(api_key=api_key)

    def encode_image(self, image_path):
        """
        Encode an image to base64.

        Args:
            image_path (str): Path to the image file

        Returns:
            str or None: Base64 encoded string or None if encoding fails
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def call(self, image_path, include_images=True):
        """
        Process an image through OCR.

        Args:
            image_path (str): Path to the image file
            include_images (bool, optional): Whether to request base64 image data. Defaults to True.

        Returns:
            object or None: OCR response object or None if processing fails
        """
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            if base64_image is None:
                return None

            # Determine image type from file extension
            image_type = Path(image_path).suffix.lstrip(".")
            if not image_type:
                image_type = "jpeg"  # Default if no extension

            # Process the image
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/{image_type};base64,{base64_image}",
                },
                include_image_base64=include_images,
            )

            return ocr_response
        except Exception as e:
            print(f"Error processing OCR: {e}")
            return None
