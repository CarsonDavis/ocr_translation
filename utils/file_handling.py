# utils/file_handling.py
import os
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

    This function ensures consistent behavior whether output_dir is a directory
    or a file path.

    Args:
        input_path (str): Path to the input file
        output_dir (str): Directory for output (or file path if it has an extension)
        suffix (str, optional): Suffix to add to filename. Defaults to None.

    Returns:
        str: Generated output path
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_dir)

    # If output_dir is None, use current directory
    if output_dir is None:
        output_dir = "."
        output_path_obj = Path(output_dir)

    # If output_dir is an existing directory, treat it as a directory
    if output_path_obj.is_dir():
        # Create full output path within the directory
        base_name = input_path_obj.stem
        if suffix:
            base_name = f"{base_name}_{suffix}"
        return str(output_path_obj / f"{base_name}{input_path_obj.suffix}")

    # If output_dir has a file extension AND parent directory exists or is creatable,
    # treat it as a specific file path
    if output_path_obj.suffix and (
        output_path_obj.parent.exists() or output_path_obj.parent == Path(".")
    ):
        # Ensure parent directory exists
        ensure_dir(output_path_obj.parent)
        return str(output_path_obj)

    # Otherwise treat as directory name and create it
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
