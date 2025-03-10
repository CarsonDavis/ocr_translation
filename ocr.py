"""
ocr.py - Script to perform OCR on document images
"""

import os
import re
import base64
import argparse
from pathlib import Path
from typing import Any, Tuple

# Import from utility module
import utils.utils as utils
from utils import file_handling
from utils.constants import (
    DEFAULT_OCR_DIR,
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_OCR_MODEL,
    DEFAULT_PROCESS_IMAGES,
)


def save_extracted_images(
    page: Any, base_document_name: str, output_dir: str = DEFAULT_OCR_DIR
) -> Tuple[list[str], dict[str, str]]:
    """
    Save extracted images from OCR response to a document-specific directory.

    Args:
        page: Page object from OCR response
        base_document_name: Base name of the document (without extension)
        output_dir: Base directory for output

    Returns:
        Tuple containing:
        - List of saved image paths
        - Dictionary mapping original image IDs to new relative paths
    """
    saved_images: list[str] = []
    image_path_mapping: dict[str, str] = {}

    # Create document-specific images directory
    images_dir = os.path.join(output_dir, "images", base_document_name)
    file_handling.ensure_dir(images_dir)

    # Process each image in the page
    for img_obj in page.images:
        try:
            # Use image ID as filename
            original_id: str = img_obj.id
            filename: str = f"{original_id}"

            # If the filename doesn't have an extension, add .jpg
            if not Path(filename).suffix:
                filename = f"{filename}.jpg"

            output_path: str = os.path.join(images_dir, filename)

            # Create the relative path for markdown references
            relative_path: str = os.path.join("images", base_document_name, filename)
            image_path_mapping[original_id] = relative_path

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
                print(f"No base64 data available for image {original_id}")
        except Exception as e:
            print(f"Error saving image {img_obj.id}: {e}")

    return saved_images, image_path_mapping


def update_image_links_in_markdown(
    markdown_text: str, image_path_mapping: dict[str, str]
) -> str:
    """
    Update image references in markdown to point to the new image locations.

    Args:
        markdown_text: Original markdown text
        image_path_mapping: Dictionary mapping original image IDs to new relative paths

    Returns:
        str: Updated markdown text with corrected image references
    """
    updated_text: str = markdown_text

    # Replace markdown image links: ![alt](image_id) -> ![alt](new_path)
    for original_id, new_path in image_path_mapping.items():
        # Escape special characters in the ID for regex
        escaped_id: str = re.escape(original_id)

        # Match markdown image syntax and replace with updated path
        updated_text = re.sub(
            rf"!\[(.*?)\]\({escaped_id}\)", rf"![\1]({new_path})", updated_text
        )

        # Also handle HTML image tags with this ID
        updated_text = re.sub(
            rf'<img([^>]*?)src=["\']{escaped_id}["\']([^>]*?)>',
            rf'<img\1src="{new_path}"\2>',
            updated_text,
        )

    return updated_text


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
    output_dir: str = DEFAULT_OCR_DIR,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    api_key: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
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

    # Create output directory if needed
    file_handling.ensure_dir(output_dir)

    # Create OCR client and process image
    ocr_client = utils.OCRAI(api_key=api_key, model=model)
    ocr_response = ocr_client.call(image_path, include_images=process_images)

    if ocr_response is None:
        return results

    # Get markdown from first page
    if ocr_response.pages and len(ocr_response.pages) > 0:
        markdown_text: str = ocr_response.pages[0].markdown

        # Generate base name for the document
        base_name: str = os.path.splitext(os.path.basename(image_path))[0]

        # Process images if requested
        if process_images:
            # Save extracted images to document-specific directory and get path mapping
            saved_images, image_path_mapping = save_extracted_images(
                ocr_response.pages[0], base_name, output_dir
            )
            results["image_paths"] = saved_images

            # Update image links in the markdown to point to the new image locations
            markdown_text = update_image_links_in_markdown(
                markdown_text, image_path_mapping
            )
        else:
            # Clean markdown to remove image references if images are not processed
            markdown_text = remove_images_from_markdown(
                markdown_text, ocr_response.pages[0]
            )

        # Use get_output_path to ensure consistent behavior
        markdown_path: str = file_handling.get_output_path(
            f"{base_name}.md", output_dir  # Create a fake input path with .md extension
        )

        # Save markdown
        if file_handling.save_markdown(markdown_text, markdown_path):
            results["markdown_path"] = markdown_path
            results["success"] = True

    return results


def process_batch(
    input_dir: str,
    output_dir: str = DEFAULT_OCR_DIR,
    file_pattern: str = DEFAULT_IMAGE_PATTERN,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    api_key: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
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
        image_files.extend(file_handling.find_files(input_dir, pattern))

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
        default=DEFAULT_OCR_DIR,
        help=f"Directory to save output (default: '{DEFAULT_OCR_DIR}'). "
        f"Files will be saved within this directory with names derived from the input files.",
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
        default=DEFAULT_IMAGE_PATTERN,
        help=f"File pattern when using batch mode (default: '{DEFAULT_IMAGE_PATTERN}')",
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
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model to use (default: '{DEFAULT_OCR_MODEL}')",
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
