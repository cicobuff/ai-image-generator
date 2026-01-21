#!/usr/bin/env python3
"""Entry point for the AI Image Generator application."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main entry point."""
    from src.app.application import run_app
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
