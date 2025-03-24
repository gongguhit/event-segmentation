#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check DV module API
"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import dv_processing as dv
    print(f"DV module imported from dv_processing: {dv.__file__}")
    
    # Print available submodules
    print("\nAvailable submodules in dv_processing:")
    for name in dir(dv):
        if not name.startswith('_'):  # Skip private attributes
            print(f"  {name}")
    
    # Check io module
    if hasattr(dv, 'io'):
        print("\nContents of dv.io module:")
        for name in dir(dv.io):
            if not name.startswith('_'):  # Skip private attributes
                print(f"  {name}")
    
    # Check if we have AedatFile instead of AedatFileReader
    if hasattr(dv, 'AedatFile'):
        print("\nFound AedatFile class in dv module")
    
    # Check specific functions for reading AEDAT4 files
    print("\nChecking specific functions for reading AEDAT4 files:")
    for name in dir(dv):
        if 'aedat' in name.lower() or 'file' in name.lower() or 'read' in name.lower():
            print(f"  {name}")
    
except ImportError as e:
    print(f"Error importing dv_processing: {e}")
    print("Please install dv_processing with: pip install dv-processing")
    sys.exit(1) 