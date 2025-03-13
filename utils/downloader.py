import requests
import json
from PIL import Image
import io
import os
import math
import concurrent.futures
import re
import time
from tqdm import tqdm

def download_iiif_image(image_id, output_path=None, max_concurrent_requests=4, use_max_regions=True):
    """
    Download and reconstruct a full resolution image from IIIF image server
    
    Args:
        image_id (str): The IIIF image ID
        output_path (str): Path where the reconstructed image will be saved. If None, will not save.
        max_concurrent_requests (int): Maximum number of concurrent download requests
        use_max_regions (bool): Whether to use the maximum allowed region size instead of tiles
    
    Returns:
        PIL.Image: The reconstructed full-resolution image
    """
    # Get the image info
    info_url = f"{image_id}/info.json"
    print(f"Fetching image info from {info_url}")
    
    try:
        response = requests.get(info_url)
        response.raise_for_status()
        info = response.json()
    except requests.RequestException as e:
        print(f"Error fetching image info for {image_id}: {e}")
        return None
    
    # Extract dimensions
    width = info['width']
    height = info['height']
    print(f"Image dimensions: {width}x{height}")
    
    # Determine region size to use
    if use_max_regions:
        # Use the maximum allowed region size
        max_width = 2000  # Default maximum
        max_height = 2000  # Default maximum
        
        # Check if server specifies maximum dimensions
        if 'profile' in info and len(info['profile']) > 1 and isinstance(info['profile'][1], dict):
            profile_info = info['profile'][1]
            max_width = profile_info.get('maxWidth', 2000)
            max_height = profile_info.get('maxHeight', 2000)
        
        region_width = max_width
        region_height = max_height
        print(f"Using maximum region size: {region_width}x{region_height}")
    else:
        # Use the tile size from the server
        region_width = 256
        region_height = 256
        
        # Check if tile info is available
        if 'tiles' in info and len(info['tiles']) > 0:
            tile_info = info['tiles'][0]
            region_width = tile_info.get('width', 256)
            region_height = tile_info.get('height', 256)
        
        print(f"Using tile size: {region_width}x{region_height}")
    
    # Calculate the number of regions needed in each dimension
    cols = math.ceil(width / region_width)
    rows = math.ceil(height / region_height)
    
    print(f"Grid size: {cols}x{rows} regions ({cols * rows} total regions)")
    
    # Create a new image with the full dimensions
    full_image = Image.new('RGB', (width, height))
    
    def download_and_place_region(row, col):
        """Download a single region and place it in the full image"""
        x = col * region_width
        y = row * region_height
        
        # Calculate the actual width and height of this region (might be smaller at edges)
        actual_region_width = min(region_width, width - x)
        actual_region_height = min(region_height, height - y)
        
        # IIIF URL format: {id}/{region}/{size}/{rotation}/{quality}.{format}
        # region format: x,y,width,height
        region = f"{x},{y},{actual_region_width},{actual_region_height}"
        
        # We want the region at its original size
        size = "full"  # Using 'full' maintains the region's original size
        
        region_url = f"{image_id}/{region}/{size}/0/default.jpg"
        
        try:
            region_response = requests.get(region_url)
            region_response.raise_for_status()
            
            # Open the region image
            region_image = Image.open(io.BytesIO(region_response.content))
            
            # Paste the region into the full image
            full_image.paste(region_image, (x, y))
            
            return True
        except requests.RequestException as e:
            print(f"Error downloading region at ({x},{y}): {e}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading region at ({x},{y}): {e}")
            return False
    
    # Create a list of all regions to download
    tasks = [(row, col) for row in range(rows) for col in range(cols)]
    
    # Download regions in parallel
    success_count = 0
    total_regions = len(tasks)
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        future_to_task = {executor.submit(download_and_place_region, row, col): (row, col) 
                          for row, col in tasks}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_task), 1):
            task = future_to_task[future]
            success = future.result()
            if success:
                success_count += 1
            
            # Print progress
            print(f"Progress: {i}/{total_regions} regions processed ({success_count} successful)", end='\r')
    
    print(f"\nDownloaded {success_count}/{total_regions} regions successfully")
    
    # Save the reconstructed image if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"Saving image to {output_path}")
        full_image.save(output_path, quality=95)
    
    return full_image

def download_iiif_sequence(base_url, start_page=1, end_page=5, output_dir="iiif_images", page_digit_padding=5, delay_between_images=2, format_string="{base}-{page}.jp2"):
    """
    Download a sequence of IIIF images
    
    Args:
        base_url (str): Base URL of the IIIF images, without the page number and extension
        start_page (int): First page number
        end_page (int): Last page number (inclusive)
        output_dir (str): Directory where images will be saved
        page_digit_padding (int): Number of digits to pad the page number with zeros
        delay_between_images (int): Delay in seconds between downloading images
        format_string (str): Format string for constructing the URL. Use {base} for base_url and {page} for page number
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to directory: {output_dir}")
    
    # Ensure base_url doesn't have trailing characters that would interfere with formatting
    base_url = base_url.rstrip('-.')
    
    # Download each image
    successful_images = 0
    total_images = end_page - start_page + 1
    
    for page_num in range(start_page, end_page + 1):
        # Format with leading zeros based on the specified padding
        page_str = f"{page_num:0{page_digit_padding}d}"
        
        # Construct the image ID using the format string
        image_id = format_string.format(base=base_url, page=page_str)
        
        output_filename = f"{page_str}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n==== Processing image {page_num} of {end_page} ({page_num-start_page+1}/{total_images}) ====")
        print(f"URL: {image_id}")
        
        try:
            image = download_iiif_image(
                image_id, 
                output_path, 
                max_concurrent_requests=4,
                use_max_regions=True
            )
            
            if image:
                successful_images += 1
                print(f"Successfully downloaded image {page_num} ({successful_images}/{total_images})")
            else:
                print(f"Failed to download image {page_num}")
                
            # Add a small delay between images to avoid overloading the server
            if page_num < end_page:
                print(f"Waiting {delay_between_images} seconds before next image...")
                time.sleep(delay_between_images)
                
        except Exception as e:
            print(f"Error processing image {page_num}: {e}")
    
    print(f"\n==== Summary ====")
    print(f"Completed downloading {successful_images} out of {total_images} images")
    print(f"Images saved to {output_dir}")

def download_single_image(image_id, output_path="single_image.jpg"):
    """Download a single IIIF image"""
    print(f"Downloading single image from {image_id}")
    download_iiif_image(image_id, output_path, use_max_regions=True)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Example 1: Download Montaigne manuscript pages from Cambridge University Library
    download_iiif_sequence(
        base_url="https://images.lib.cam.ac.uk/iiif/PR-MONTAIGNE-00001-00007-00022-000",
        start_page=22,
        end_page=182,
        output_dir="downloaded_images",
        page_digit_padding=5,  # Pages are formatted as 00001, 00002, etc.
        delay_between_images=0.25,
        format_string="{base}-{page}.jp2"  # Format: base-00001.jp2
    )
    
    # Example 2: Generic usage template for a different URL format (commented out)
    # download_iiif_sequence(
    #     base_url="https://example.org/iiif/manuscript",
    #     start_page=1,
    #     end_page=10,
    #     output_dir="output_images",
    #     page_digit_padding=3,  # Pages formatted as 001, 002, etc.
    #     delay_between_images=1,
    #     format_string="{base}/{page}/info.jp2"  # Different format pattern
    # )
    
    # Example 3: Download a single image
    # download_single_image(
    #     "https://images.lib.cam.ac.uk/iiif/PR-MONTAIGNE-00001-00007-00022-000-00007.jp2",
    #     "montaigne_page7.jpg"
    # )