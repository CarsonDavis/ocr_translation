#!/usr/bin/env python3
"""
ocr.py - Script to perform OCR on document images
"""

import os
import re
import base64
import argparse
from pathlib import Path
from typing import Any

# Import from utility module
import utils.utils as utils
from utils import file_handling

# Default values as constants - single source of truth
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_IMAGES_DIR = "images"
DEFAULT_FILE_PATTERN = "*.jpg *.png *.jpeg *.webp"
DEFAULT_MODEL = "mistral-ocr-latest"
DEFAULT_PROCESS_IMAGES = True


def save_extracted_images(page: Any, output_dir: str = DEFAULT_IMAGES_DIR) -> list[str]:
    """
    Save extracted images from OCR response.

    Args:
        page: Page object from OCR response
        output_dir: Directory to save images to

    Returns:
        list: List of saved image paths
    """
    saved_images: list[str] = []

    # Create output directory if it doesn't exist
    file_handling.ensure_dir(output_dir)

    # Process each image in the page
    for img_obj in page.images:
        try:
            # Use image ID as filename
            filename: str = f"{img_obj.id}"
            output_path: str = os.path.join(output_dir, filename)

            # Check if image_base64 is available
            if img_obj.image_base64:
                # Remove the data URL prefix if present
                base64_str: str = img_obj.image_base64
                prefix: str = "data:image/jpeg;base64,"
                if base64_str.startswith(prefix):
                    base64_str = base64_str[len(prefix) :]

                # Decode the base64 string
                img_data: bytes = base64.b64decode(base64_str)

                # Write the decoded image to a file
                with open(output_path, "wb") as f:
                    f.write(img_data)

                saved_images.append(output_path)
                print(f"Saved image to {output_path}")
            else:
                print(f"No base64 data available for image {img_obj.id}")
        except Exception as e:
            print(f"Error saving image {img_obj.id}: {e}")

    return saved_images


def remove_images_from_markdown(markdown_text: str, page: Any) -> str:
    """
    Remove specific image references from markdown based on image IDs in the OCR response.

    Args:
        markdown_text: Original markdown text
        page: Page object from OCR response containing images

    Returns:
        str: Cleaned markdown text without image references
    """
    # Get all image IDs from the page
    image_ids: list[str] = [img.id for img in page.images]

    cleaned_text: str = markdown_text

    # Process each image ID and remove its references
    for img_id in image_ids:
        # Escape special characters in the ID for regex
        escaped_id: str = re.escape(img_id)

        # Remove markdown image syntax (![alt](image_id))
        cleaned_text = re.sub(rf"!\[.*?\]\({escaped_id}\)", "", cleaned_text)

        # Remove HTML image tags with this ID
        cleaned_text = re.sub(rf"<img[^>]*{escaped_id}[^>]*>", "", cleaned_text)

    # Remove empty lines that might be left after removing images
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)

    return cleaned_text


def process_document(
    image_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, bool | str | None | list[str]]:
    """
    Process a document image through OCR and save the results.

    Args:
        image_path: Path to the image file
        output_dir: Directory to save output to
        process_images: Whether to process and save images
        api_key: Mistral API key. Defaults to environment variable
        model: OCR model to use

    Returns:
        dict: Results dictionary with paths to saved files
    """
    results: dict[str, bool | str | None | list[str]] = {
        "success": False,
        "markdown_path": None,
        "image_paths": [],
    }

    # Create output directory
    file_handling.ensure_dir(output_dir)

    # Create OCR client and process image
    ocr_client = utils.OCRAI(api_key=api_key, model=model)
    ocr_response = ocr_client.call(image_path, include_images=process_images)

    if ocr_response is None:
        return results

    # Get markdown from first page
    if ocr_response.pages and len(ocr_response.pages) > 0:
        markdown_text: str = ocr_response.pages[0].markdown

        # Process images if requested
        if process_images:
            # Save extracted images
            results["image_paths"] = save_extracted_images(
                ocr_response.pages[0], output_dir
            )
        else:
            # Clean markdown to remove image references
            markdown_text = remove_images_from_markdown(
                markdown_text, ocr_response.pages[0]
            )

        # Generate output filename based on input image name
        base_name: str = os.path.splitext(os.path.basename(image_path))[0]
        markdown_path: str = os.path.join(output_dir, f"{base_name}.md")

        # Save markdown
        if file_handling.save_markdown(markdown_text, markdown_path):
            results["markdown_path"] = markdown_path
            results["success"] = True

    return results


def process_batch(
    input_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    file_pattern: str = DEFAULT_FILE_PATTERN,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, bool | str | None | list[str]]]:
    """
    Process all image files in a directory.

    Args:
        input_dir: Directory containing image files to process
        output_dir: Directory to save output
        file_pattern: Pattern to match files
        process_images: Whether to process and save images
        api_key: Mistral API key. Defaults to environment variable
        model: OCR model to use

    Returns:
        list: List of results dictionaries for each file
    """
    results: list[dict[str, bool | str | None | list[str]]] = []

    # Find all image files
    image_files: list[Path] = []
    for pattern in file_pattern.split():
        image_files.extend(utils.find_files(input_dir, pattern))

    if not image_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(image_files)} image files to process")

    # Process each file
    for image_path in image_files:
        print(f"Processing {image_path}...")
        file_result = process_document(
            str(image_path),
            output_dir=output_dir,
            process_images=process_images,
            api_key=api_key,
            model=model,
        )
        results.append(file_result)

    # Print summary
    utils.print_batch_summary(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document images with OCR")

    # Input and output options
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save output (default: '{DEFAULT_OUTPUT_DIR}')",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process all image files in input directory",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default=DEFAULT_FILE_PATTERN,
        help=f"File pattern when using batch mode (default: '{DEFAULT_FILE_PATTERN}')",
    )

    # Processing options
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't process and save images (also doesn't request image data from API)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"OCR model to use (default: '{DEFAULT_MODEL}')",
    )

    args = parser.parse_args()

    # Determine if we're processing a single file or a directory
    if args.batch or os.path.isdir(args.input):
        process_batch(
            args.input,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            process_images=not args.no_images,
            model=args.model,
        )
    else:
        result = process_document(
            args.input,
            output_dir=args.output_dir,
            process_images=not args.no_images,
            model=args.model,
        )

        if result["success"]:
            print(f"Successfully processed {args.input}")
            print(f"Markdown saved to {result['markdown_path']}")
            if result["image_paths"]:
                print(f"Saved {len(result['image_paths'])} images")
        else:
            print(f"Failed to process {args.input}")
