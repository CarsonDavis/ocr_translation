"""
process.py - High-level pipeline for OCR, cleaning, and translation
"""

import os
import argparse
from pathlib import Path
from typing import Any

# Import from project modules
import ocr
import clean
import translate
import utils.utils as utils
from utils import file_handling
from utils.constants import (
    DEFAULT_OCR_SUBDIR,
    DEFAULT_CLEANED_SUBDIR,
    DEFAULT_TRANSLATED_SUBDIR,
    DEFAULT_TEMPERATURE,
    DEFAULT_PIPELINE_DIR,
    DEFAULT_SKIP_CLEAN,
    DEFAULT_SKIP_TRANSLATE,
    DEFAULT_IMAGE_PATTERN,
    DEFAULT_OCR_MODEL,
    DEFAULT_CLEAN_MODEL,
    DEFAULT_TRANSLATE_MODEL,
    DEFAULT_PROCESS_IMAGES,
    AVAILABLE_MODELS,
)


def process_image_pipeline(
    image_path: str,
    output_dir: str = DEFAULT_PIPELINE_DIR,
    ocr_model: str = DEFAULT_OCR_MODEL,
    clean_model: str = DEFAULT_CLEAN_MODEL,
    translate_model: str = DEFAULT_TRANSLATE_MODEL,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    skip_clean: bool = DEFAULT_SKIP_CLEAN,
    skip_translate: bool = DEFAULT_SKIP_TRANSLATE,
    temperature: float = DEFAULT_TEMPERATURE,  # Use the same temperature for both clean and translate
) -> dict[str, bool | str | None]:
    """
    Run the full pipeline (OCR -> Clean -> Translate) on a single image.

    Args:
        image_path: Path to the input image
        output_dir: Base directory for output
        ocr_model: OCR model to use
        clean_model: Model for cleaning
        translate_model: Model for translation
        process_images: Whether to save extracted images
        skip_clean: Skip the cleaning step
        skip_translate: Skip the translation step
        temperature: Temperature for AI generation

    Returns:
        dict: Results of the pipeline
    """
    results: dict[str, bool | str | None] = {
        "image_path": image_path,
        "ocr_success": False,
        "clean_success": False,
        "translate_success": False,
        "ocr_output": None,
        "clean_output": None,
        "translate_output": None,
    }

    # Create main output directory
    file_handling.ensure_dir(output_dir)

    # Step 1: OCR
    print(f"\n=== Step 1: OCR Processing for {image_path} ===")

    # Create subdirectory for OCR output
    ocr_dir = os.path.join(output_dir, DEFAULT_OCR_SUBDIR)
    file_handling.ensure_dir(ocr_dir)

    # Process document
    ocr_result = ocr.process_document(
        image_path, output_dir=ocr_dir, process_images=process_images, model=ocr_model
    )

    results["ocr_success"] = ocr_result["success"]
    results["ocr_output"] = ocr_result["markdown_path"]

    if not ocr_result["success"]:
        print("OCR processing failed. Pipeline stopped.")
        return results

    # Step 2: Clean (optional)
    current_input = ocr_result["markdown_path"]

    if not skip_clean:
        print(f"\n=== Step 2: Cleaning OCR output ===")

        # Create subdirectory for cleaned output
        clean_dir = os.path.join(output_dir, DEFAULT_CLEANED_SUBDIR)
        file_handling.ensure_dir(clean_dir)

        clean_output_path = file_handling.get_output_path(current_input, clean_dir)

        clean_result = clean.clean_markdown_with_llm(
            current_input, clean_output_path, model=clean_model, temperature=temperature
        )

        results["clean_success"] = clean_result["success"]
        results["clean_output"] = clean_result["output_path"]

        if clean_result["success"]:
            current_input = clean_result["output_path"]
        else:
            print("Cleaning failed, but continuing with original OCR output.")
    else:
        print("\n=== Step 2: Cleaning skipped ===")

    # Step 3: Translate (optional)
    if not skip_translate:
        print(f"\n=== Step 3: Translating to English ===")

        # Create subdirectory for translated output
        translate_dir = os.path.join(output_dir, DEFAULT_TRANSLATED_SUBDIR)
        file_handling.ensure_dir(translate_dir)

        translate_output_path = file_handling.get_output_path(
            current_input, translate_dir
        )

        translate_result = translate.translate_markdown(
            current_input,
            translate_output_path,
            model=translate_model,
            temperature=temperature,
        )

        results["translate_success"] = translate_result["success"]
        results["translate_output"] = translate_result["output_path"]
    else:
        print("\n=== Step 3: Translation skipped ===")

    # Print summary
    print("\n=== Pipeline Summary ===")
    print(f"OCR: {'Success' if results['ocr_success'] else 'Failed'}")
    if not skip_clean:
        print(f"Clean: {'Success' if results['clean_success'] else 'Failed'}")
    else:
        print("Clean: Skipped")
    if not skip_translate:
        print(f"Translate: {'Success' if results['translate_success'] else 'Failed'}")
    else:
        print("Translate: Skipped")

    return results


def process_batch_pipeline(
    input_dir: str,
    output_dir: str = DEFAULT_PIPELINE_DIR,
    file_pattern: str = DEFAULT_IMAGE_PATTERN,
    ocr_model: str = DEFAULT_OCR_MODEL,
    clean_model: str = DEFAULT_CLEAN_MODEL,
    translate_model: str = DEFAULT_TRANSLATE_MODEL,
    process_images: bool = DEFAULT_PROCESS_IMAGES,
    skip_clean: bool = DEFAULT_SKIP_CLEAN,
    skip_translate: bool = DEFAULT_SKIP_TRANSLATE,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict[str, bool | str | None]]:
    """
    Run the full pipeline on a batch of images.

    Args:
        input_dir: Directory containing input images
        output_dir: Base directory for output
        file_pattern: Pattern to match files
        ocr_model: OCR model to use
        clean_model: Model for cleaning
        translate_model: Model for translation
        process_images: Whether to save extracted images
        skip_clean: Skip the cleaning step
        skip_translate: Skip the translation step
        temperature: Temperature for AI generation

    Returns:
        list: Results for each processed image
    """
    results: list[dict[str, bool | str | None]] = []

    # Find all image files
    image_files: list[Path] = []
    for pattern in file_pattern.split():
        image_files.extend(utils.find_files(input_dir, pattern))

    if not image_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(image_files)} image files to process")

    # Process each file through the pipeline
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing {image_path}...")

        # Process in the common output directory
        result = process_image_pipeline(
            str(image_path),
            output_dir=output_dir,
            ocr_model=ocr_model,
            clean_model=clean_model,
            translate_model=translate_model,
            process_images=process_images,
            skip_clean=skip_clean,
            skip_translate=skip_translate,
            temperature=temperature,
        )

        results.append(result)

    # Print batch summary
    print("\n=== Batch Processing Summary ===")
    print(f"Total images processed: {len(results)}")
    print(
        f"OCR successful: {sum(1 for r in results if r['ocr_success'])}/{len(results)}"
    )
    if not skip_clean:
        print(
            f"Cleaning successful: {sum(1 for r in results if r['clean_success'])}/{len(results)}"
        )
    if not skip_translate:
        print(
            f"Translation successful: {sum(1 for r in results if r['translate_success'])}/{len(results)}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OCR, cleaning, and translation pipeline on document images"
    )

    # Input and output options
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_PIPELINE_DIR,
        help=f"Base directory for output (default: '{DEFAULT_PIPELINE_DIR}'). "
        f"Files will be organized in subdirectories for each processing stage.",
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

    # Pipeline control options
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip the cleaning step in the pipeline",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip the translation step in the pipeline",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't save extracted images from OCR",
    )

    # Model options
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model to use (default: '{DEFAULT_OCR_MODEL}')",
    )
    parser.add_argument(
        "--clean-model",
        default=DEFAULT_CLEAN_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"AI model to use for cleaning (default: '{DEFAULT_CLEAN_MODEL}')",
    )
    parser.add_argument(
        "--translate-model",
        default=DEFAULT_TRANSLATE_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"AI model to use for translation (default: '{DEFAULT_TRANSLATE_MODEL}')",
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
        process_batch_pipeline(
            args.input,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            ocr_model=args.ocr_model,
            clean_model=args.clean_model,
            translate_model=args.translate_model,
            process_images=not args.no_images,
            skip_clean=args.skip_clean,
            skip_translate=args.skip_translate,
            temperature=args.temperature,
        )
    else:
        process_image_pipeline(
            args.input,
            output_dir=args.output_dir,
            ocr_model=args.ocr_model,
            clean_model=args.clean_model,
            translate_model=args.translate_model,
            process_images=not args.no_images,
            skip_clean=args.skip_clean,
            skip_translate=args.skip_translate,
            temperature=args.temperature,
        )
