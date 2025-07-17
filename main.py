#!/usr/bin/env python3
"""
Main entry point for Meta-Learning Pseudo-Labels Community Detection

Usage:
    python main.py --help                    # Show help
    python main.py --quick                   # Quick test run
    python main.py --step data               # Run only data preparation
    python main.py --step train              # Run only training
    python main.py                           # Run full pipeline
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments import main

if __name__ == "__main__":
    main()
