# full_ocr.py
import base64
import os
import re
import requests
from pathlib import Path
from mistralai import Mistral


def encode_image(image_path):
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


def process_ocr(
    image_path, api_key=None, model="mistral-ocr-latest", include_images=True
):
    """
    Process an image through Mistral's OCR API.

    Args:
        image_path (str): Path to the image file
        api_key (str, optional): Mistral API key. Defaults to environment variable.
        model (str, optional): OCR model to use. Defaults to "mistral-ocr-latest".
        include_images (bool, optional): Whether to request base64 image data. Defaults to True.

    Returns:
        object or None: OCR response object or None if processing fails
    """
    try:
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided and MISTRAL_API_KEY environment variable not set"
                )

        # Encode the image
        base64_image = encode_image(image_path)
        if base64_image is None:
            return None

        # Determine image type from file extension
        image_type = Path(image_path).suffix.lstrip(".")
        if not image_type:
            image_type = "jpeg"  # Default if no extension

        # Create Mistral client and process the image
        client = Mistral(api_key=api_key)
        ocr_response = client.ocr.process(
            model=model,
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


def save_extracted_images(page, output_dir="images"):
    """
    Save extracted images from OCR response.

    Args:
        page: Page object from OCR response
        output_dir (str, optional): Directory to save images to. Defaults to "images".

    Returns:
        list: List of saved image paths
    """
    saved_images = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the page
    for img_obj in page.images:
        try:
            # Use image ID as filename
            filename = f"{img_obj.id}"
            output_path = os.path.join(output_dir, filename)

            # Check if image_base64 is available
            if img_obj.image_base64:
                # Remove the data URL prefix if present
                base64_str = img_obj.image_base64
                prefix = "data:image/jpeg;base64,"
                if base64_str.startswith(prefix):
                    base64_str = base64_str[len(prefix) :]

                # Decode the base64 string
                img_data = base64.b64decode(base64_str)

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


def clean_markdown(markdown_text, page):
    """
    Remove specific image references from markdown based on image IDs in the OCR response.

    Args:
        markdown_text (str): Original markdown text
        page: Page object from OCR response containing images

    Returns:
        str: Cleaned markdown text without image references
    """
    # Get all image IDs from the page
    image_ids = [img.id for img in page.images]

    cleaned_text = markdown_text

    # Process each image ID and remove its references
    for img_id in image_ids:
        # Escape special characters in the ID for regex
        escaped_id = re.escape(img_id)

        # Remove markdown image syntax (![alt](image_id))
        cleaned_text = re.sub(rf"!\[.*?\]\({escaped_id}\)", "", cleaned_text)

        # Remove HTML image tags with this ID
        cleaned_text = re.sub(rf"<img[^>]*{escaped_id}[^>]*>", "", cleaned_text)

    # Remove empty lines that might be left after removing images
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)

    return cleaned_text


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


def process_document(
    image_path, output_dir="output", process_images=True, api_key=None
):
    """
    Process a document image through OCR and save the results.

    Args:
        image_path (str): Path to the image file
        output_dir (str, optional): Directory to save output to. Defaults to "output".
        process_images (bool, optional): Whether to process and save images. Defaults to True.
                                        When False, image data won't be requested from the API.
        api_key (str, optional): Mistral API key. Defaults to environment variable.

    Returns:
        dict: Results dictionary with paths to saved files
    """
    results = {"success": False, "markdown_path": None, "image_paths": []}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process OCR - only request image data if we're going to use it
    ocr_response = process_ocr(image_path, api_key, include_images=process_images)
    if ocr_response is None:
        return results

    # Get markdown from first page
    if ocr_response.pages and len(ocr_response.pages) > 0:
        markdown_text = ocr_response.pages[0].markdown

        # Process images if requested
        if process_images:
            # Save extracted images
            # image_dir = os.path.join(output_dir, "images")
            results["image_paths"] = save_extracted_images(
                ocr_response.pages[0], output_dir
            )
        else:
            # Clean markdown to remove image references
            markdown_text = clean_markdown(markdown_text, ocr_response.pages[0])

        # Generate output filename based on input image name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        markdown_path = os.path.join(output_dir, f"{base_name}.md")

        # Save markdown
        if save_markdown(markdown_text, markdown_path):
            results["markdown_path"] = markdown_path
            results["success"] = True

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process document images with OCR")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument(
        "--output-dir", default="output", help="Directory to save output"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't process and save images (also doesn't request image data from API)",
    )

    args = parser.parse_args()

    result = process_document(
        args.image_path, output_dir=args.output_dir, process_images=not args.no_images
    )

    if result["success"]:
        print(f"Successfully processed {args.image_path}")
        print(f"Markdown saved to {result['markdown_path']}")
        if result["image_paths"]:
            print(f"Saved {len(result['image_paths'])} images")
    else:
        print(f"Failed to process {args.image_path}")
