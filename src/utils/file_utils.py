"""File utility functions."""

from pathlib import Path
from typing import Optional, Generator
import shutil
import hashlib

from src.utils.constants import MODEL_EXTENSIONS


def get_file_hash(path: Path, algorithm: str = "sha256") -> Optional[str]:
    """
    Calculate hash of a file.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex digest string or None if file doesn't exist
    """
    if not path.exists():
        return None

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size_str(size_bytes: int) -> str:
    """
    Convert file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def find_model_files(directory: Path) -> Generator[Path, None, None]:
    """
    Find all model files in a directory recursively.

    Args:
        directory: Directory to search

    Yields:
        Paths to model files
    """
    if not directory.exists():
        return

    for ext in MODEL_EXTENSIONS:
        yield from directory.rglob(f"*{ext}")


def ensure_directory(path: Path) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        True if directory exists or was created successfully
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False


def safe_delete(path: Path) -> bool:
    """
    Safely delete a file or directory.

    Args:
        path: Path to delete

    Returns:
        True if deletion was successful
    """
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"Error deleting {path}: {e}")
        return False


def get_unique_filename(directory: Path, basename: str, extension: str) -> Path:
    """
    Get a unique filename in a directory by appending numbers if needed.

    Args:
        directory: Target directory
        basename: Base name for the file
        extension: File extension (including dot)

    Returns:
        Unique path in the directory
    """
    path = directory / f"{basename}{extension}"
    counter = 1

    while path.exists():
        path = directory / f"{basename}_{counter}{extension}"
        counter += 1

    return path


def copy_file_with_progress(
    src: Path,
    dst: Path,
    progress_callback: Optional[callable] = None,
    chunk_size: int = 1024 * 1024,  # 1MB chunks
) -> bool:
    """
    Copy a file with progress reporting.

    Args:
        src: Source file path
        dst: Destination file path
        progress_callback: Optional callback(bytes_copied, total_bytes)
        chunk_size: Size of chunks to copy

    Returns:
        True if copy was successful
    """
    try:
        total_size = src.stat().st_size
        copied = 0

        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                if progress_callback:
                    progress_callback(copied, total_size)

        return True
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def is_valid_image(path: Path) -> bool:
    """
    Check if a file is a valid image.

    Args:
        path: Path to check

    Returns:
        True if file is a valid image
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    return path.is_file() and path.suffix.lower() in valid_extensions


def is_valid_model(path: Path) -> bool:
    """
    Check if a file is a valid model file.

    Args:
        path: Path to check

    Returns:
        True if file is a valid model
    """
    return path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
