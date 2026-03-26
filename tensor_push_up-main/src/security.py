"""
Security Module

This module provides security validation functions for the Tensor Push Up project.
It includes path validation, file size limits, extension validation, and input sanitization.
"""

import os
import re
from typing import List, Optional, Set


# Constants for security validation
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
MAX_MODEL_SIZE = 100 * 1024 * 1024  # 100MB
MAX_CONFIG_SIZE = 1024 * 1024  # 1MB
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
ALLOWED_MODEL_EXTENSIONS = {'.keras', '.h5'}
ALLOWED_CONFIG_EXTENSIONS = {'.yaml', '.yml'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def validate_file_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
    """
    Validate file path to prevent directory traversal attacks.

    Args:
        file_path: Path to validate
        allowed_dirs: List of allowed directories (if None, current directory is used)

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)

        # If no allowed directories specified, use current directory
        if allowed_dirs is None:
            allowed_dirs = [os.path.abspath(os.getcwd())]

        # Check if path is within allowed directories
        return any(abs_path.startswith(os.path.abspath(d)) for d in allowed_dirs)
    except (AttributeError, TypeError, OSError):
        return False


def validate_file_size(file_path: str, max_size: int = MAX_VIDEO_SIZE) -> bool:
    """
    Validate file size to prevent DoS attacks.

    Args:
        file_path: Path to file
        max_size: Maximum allowed file size in bytes

    Returns:
        True if file size is acceptable, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        return file_size <= max_size
    except (OSError, TypeError):
        return False


def validate_file_extension(file_path: str, allowed_extensions: Set[str]) -> bool:
    """
    Validate file extension.

    Args:
        file_path: Path to file
        allowed_extensions: Set of allowed extensions

    Returns:
        True if extension is allowed, False otherwise
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in allowed_extensions
    except (AttributeError, TypeError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace potentially dangerous characters
    sanitized = re.sub(r'[^\w\-_\.]', '', filename)

    # Prevent hidden files
    if sanitized.startswith('.'):
        sanitized = 'file_' + sanitized

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext

    return sanitized


def sanitize_input(input_str: str) -> str:
    """
    Sanitize string input to prevent injection attacks.

    Args:
        input_str: Input string to sanitize

    Returns:
        Sanitized string
    """
    if not isinstance(input_str, str):
        return str(input_str)

    # Remove control characters except basic whitespace
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)

    # Limit length to prevent memory issues
    if len(sanitized) > 1000:
        sanitized = sanitized[:1000]

    return sanitized


def validate_video_file(file_path: str) -> bool:
    """
    Validate video file with all security checks.

    Args:
        file_path: Path to video file

    Returns:
        True if video file is valid and safe, False otherwise
    """
    return (
        validate_file_path(file_path) and
        validate_file_extension(file_path, ALLOWED_VIDEO_EXTENSIONS) and
        validate_file_size(file_path, MAX_VIDEO_SIZE)
    )


def validate_model_file(file_path: str) -> bool:
    """
    Validate model file with all security checks.

    Args:
        file_path: Path to model file

    Returns:
        True if model file is valid and safe, False otherwise
    """
    return (
        validate_file_path(file_path) and
        validate_file_extension(file_path, ALLOWED_MODEL_EXTENSIONS) and
        validate_file_size(file_path, MAX_MODEL_SIZE)
    )


def validate_config_file(file_path: str) -> bool:
    """
    Validate configuration file with all security checks.

    Args:
        file_path: Path to config file

    Returns:
        True if config file is valid and safe, False otherwise
    """
    return (
        validate_file_path(file_path) and
        validate_file_extension(file_path, ALLOWED_CONFIG_EXTENSIONS) and
        validate_file_size(file_path, MAX_CONFIG_SIZE)
    )


class SecurityValidator:
    """Context manager for batch security validation."""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        """
        Initialize security validator.

        Args:
            allowed_dirs: List of allowed directories
        """
        self.allowed_dirs = allowed_dirs or [os.getcwd()]

    def validate_path(self, file_path: str) -> bool:
        """Validate file path within context."""
        return validate_file_path(file_path, self.allowed_dirs)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""


def get_security_report(file_path: str, file_type: str = 'video') -> dict:
    """
    Generate a security report for a file.

    Args:
        file_path: Path to file
        file_type: Type of file ('video', 'model', 'config')

    Returns:
        Dictionary with security validation results
    """
    report = {
        'file_path': file_path,
        'file_exists': os.path.exists(file_path),
        'valid_path': False,
        'valid_extension': False,
        'valid_size': False,
        'overall_valid': False
    }

    if not report['file_exists']:
        return report

    # Set file-specific parameters
    if file_type == 'video':
        max_size = MAX_VIDEO_SIZE
        allowed_exts = ALLOWED_VIDEO_EXTENSIONS
    elif file_type == 'model':
        max_size = MAX_MODEL_SIZE
        allowed_exts = ALLOWED_MODEL_EXTENSIONS
    elif file_type == 'config':
        max_size = MAX_CONFIG_SIZE
        allowed_exts = ALLOWED_CONFIG_EXTENSIONS
    else:
        return report

    # Perform validations
    report['valid_path'] = validate_file_path(file_path)
    report['valid_extension'] = validate_file_extension(file_path, allowed_exts)
    report['valid_size'] = validate_file_size(file_path, max_size)
    report['overall_valid'] = all([report['valid_path'], report['valid_extension'], report['valid_size']])

    # Add file details
    try:
        report['file_size'] = os.path.getsize(file_path)
        report['file_extension'] = os.path.splitext(file_path)[1].lower()
    except OSError:
        pass

    return report