#!/usr/bin/env python3
"""
Setup script for Affective State Modeling to Predict Training Dropout in Military Academies

Authors: Cindy Von Ahlefeldt, Ancuta Margondai, Mustapha Mouloua Ph.D.
Institution: University of Central Florida
Conference: MODSIM World 2025
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Filter out comments and empty lines
    requirements = [req for req in requirements if not req.startswith('#') and req.strip()]

setup(
    name="affective-state-digital-twin-simulation",
    version="1.0.0",
    description="Affective State Modeling to Predict Training Dropout in Military Academies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Cindy Von Ahlefeldt, Ancuta Margondai, Mustapha Mouloua",
    author_email="Cindy.Vonahlefeldt@ucf.edu, Ancuta.Margondai@ucf.edu, Mustapha.Mouloua@ucf.edu",
    
    # Project URLs
    url="https://github.com/yourusername/affective-state-digital-twin-simulation",
    
    # Package information
    python_requires=">=3.8",
    install_requires=requirements,
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords
    keywords=[
        "digital-twin", "affective-state", "military-training", "dropout-prediction",
        "heart-rate-variability", "HRV", "RMSSD", "markov-model", "stress-monitoring",
        "physiological-computing", "burnout-prevention"
    ],
)
