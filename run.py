#!/usr/bin/env python3
"""
Quick-start runner for the DEVIL Football Prediction Engine.

Usage:
    python run.py

Requires:
    - matches.csv in current directory with columns:
      Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals
    - Python 3.8+
    - scikit-learn, xgboost, numpy, pandas, scipy
"""

import sys
import os

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    exec(open("main.py").read())
