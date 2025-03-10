#!/usr/bin/env python3
"""
process.py - High-level pipeline for OCR, cleaning, and translation
"""

import os
import argparse
from pathlib import Path

# Import from project modules
import ocr
import clean
import translate
import utils


def process_image_pipeline(
    image_path,
    output_dir="pipeline_output",
    ocr_model="mistral-ocr-latest",
    clean_model="openai:gpt-4o",
    translate_model="openai:gpt-4o",
    process_images=True,
    skip_clean=False,
    skip_translate=False,
    temperature=0.75,
):
    """
    Run the full pipeline (OCR -> Clean -> Translate) on a single image.

    Args:
        image_path (str): Path to the input image
        output_dir (str, optional): Base directory for output. Defaults to "pipeline_output".
        ocr_model (str, optional): OCR model to use. Defaults to "mistral-ocr-latest".
        clean_model (str, optional): Model for cleaning. Defaults to "openai:gpt-4o".
        translate_model (str, optional): Model for translation. Defaults to "openai:gpt-4o".
        process_images (bool, optional): Whether to save extracted images. Defaults to True.
        skip_clean (bool, optional): Skip the cleaning step. Defaults to False.
        skip_translate (bool, optional): Skip the translation step. Defaults to False.
        temperature (float, optional): Temperature for AI generation. Defaults to 0.75.

    Returns:
        dict: Results of the pipeline
    """
    results = {
        "image_path": image_path,
        "ocr_success": False,
        "clean_success": False,
        "translate_success": False,
        "ocr_output": None,
        "clean_output": None,
        "translate_output": None,
    }

    # Create main output directory
    utils.ensure_dir(output_dir)

    # Step 1: OCR
    print(f"\n=== Step 1: OCR Processing for {image_path} ===")

    # Create subdirectory for OCR output
    ocr_dir = os.path.join(output_dir, "ocr")
    utils.ensure_dir(ocr_dir)

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
        clean_dir = os.path.join(output_dir, "cleaned")
        utils.ensure_dir(clean_dir)

        clean_output_path = utils.get_output_path(current_input, clean_dir)

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
        translate_dir = os.path.join(output_dir, "translated")
        utils.ensure_dir(translate_dir)

        translate_output_path = utils.get_output_path(current_input, translate_dir)

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
    input_dir,
    output_dir="pipeline_output",
    file_pattern="*.jpg *.png *.jpeg",
    ocr_model="mistral-ocr-latest",
    clean_model="openai:gpt-4o",
    translate_model="openai:gpt-4o",
    process_images=True,
    skip_clean=False,
    skip_translate=False,
    temperature=0.75,
):
    """
    Run the full pipeline on a batch of images.

    Args:
        input_dir (str): Directory containing input images
        output_dir (str, optional): Base directory for output. Defaults to "pipeline_output".
        file_pattern (str, optional): Pattern to match files. Defaults to "*.jpg *.png *.jpeg".
        ocr_model (str, optional): OCR model to use. Defaults to "mistral-ocr-latest".
        clean_model (str, optional): Model for cleaning. Defaults to "openai:gpt-4o".
        translate_model (str, optional): Model for translation. Defaults to "openai:gpt-4o".
        process_images (bool, optional): Whether to save extracted images. Defaults to True.
        skip_clean (bool, optional): Skip the cleaning step. Defaults to False.
        skip_translate (bool, optional): Skip the translation step. Defaults to False.
        temperature (float, optional): Temperature for AI generation. Defaults to 0.75.

    Returns:
        list: Results for each processed image
    """
    results = []

    # Find all image files
    image_files = []
    for pattern in file_pattern.split():
        image_files.extend(utils.find_files(input_dir, pattern))

    if not image_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return results

    print(f"Found {len(image_files)} image files to process")

    # Process each file through the pipeline
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing {image_path}...")

        # Create a unique output directory for each input file
        file_output_dir = os.path.join(output_dir, Path(image_path).stem)

        result = process_image_pipeline(
            str(image_path),
            output_dir=file_output_dir,
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
        default="pipeline_output",
        help="Base directory for output (default: 'pipeline_output')",
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
        default="*.jpg *.png *.jpeg",
        help="File pattern when using batch mode",
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
        "--no-images", action="store_true", help="Don't save extracted images from OCR"
    )

    # Model options
    parser.add_argument(
        "--ocr-model", default="mistral-ocr-latest", help="OCR model to use"
    )
    parser.add_argument(
        "--clean-model",
        default="openai:gpt-4o",
        choices=clean.MODELS,
        help="AI model to use for cleaning",
    )
    parser.add_argument(
        "--translate-model",
        default="openai:gpt-4o",
        choices=translate.MODELS,
        help="AI model to use for translation",
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
