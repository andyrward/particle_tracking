# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, clear_output

from detect import detect_frame
from detect_utils import track_frames
from detect_utils import estimate_minmass
from detection_ui import setup_detection_ui


# Step 1: File browser with native file dialog
import tkinter as tk
from tkinter import filedialog

# Global state
stack = None
stack_shape = None

# Create root window (hidden)
root = tk.Tk()
root.withdraw()

# Let user browse for TIFF file
file_path = filedialog.askopenfilename(
    title='Select TIFF file',
    filetypes=[('TIFF files', '*.tif *.tiff'), ('All files', '*.*')],
    initialdir='.'
)

root.destroy()

if file_path:
    try:
        stack = iio.imread(file_path)
        stack_shape = stack.shape
        
        # Calculate image statistics
        img_min = float(stack.min())
        img_max = float(stack.max())
        
        print(f'✓ Loaded: {file_path}')
        print(f'  Shape: {stack_shape}')
        print(f'  Range: [{img_min:.0f}, {img_max:.0f}]')
    except Exception as e:
        print(f'Error loading file: {str(e)}')
        stack = None
        stack_shape = None
else:
    print('No file selected. Please run this cell again to browse for a file.')


    # Set up and display detection UI
setup_detection_ui(stack, stack_shape)