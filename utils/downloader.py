"""
this script downloads a copy of Coras's book
"""

import os
import requests


def download_images(start, end, base_url, dest_folder):
    # Create the destination folder if it doesn't exist.
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for i in range(start, end + 1):
        # Format the number as a five-digit string.
        file_number = f"{i:05}"
        # Construct the full URL.
        url = f"{base_url}{file_number}.jpg"
        print(f"Downloading {url} ...")

        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(dest_folder, file_number + ".jpg")
                with open(file_path, "wb") as f:
                    # Write the image in chunks.
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"Saved to {file_path}")
            else:
                print(f"Failed to retrieve {url} (Status code: {response.status_code})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    start = 22
    end = 182
    # Base URL up to the image number part.
    base_url = "https://images.lib.cam.ac.uk//content/images/PR-MONTAIGNE-00001-00007-00022-000-"
    dest_folder = "downloaded_images"

    download_images(start, end, base_url, dest_folder)
