import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
import re

def next_log_number(folder_path):
    """
    Returns the next unused integer filename (N.log) inside folder_path.
    """
    os.makedirs(folder_path, exist_ok=True)
    pattern = re.compile(r'^(\d+)\.log$')
    used_numbers = set()

    # Iterate through folder and collect used numbers
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            used_numbers.add(int(match.group(1)))

    # Find the smallest non-used integer
    n = 0
    while n in used_numbers:
        n += 1

    return f"{folder_path}/{n}"

def continue_skip_abort():
    user_input = input("Ready for next sequence? (press Enter to continue, 'n'/'no'/'abort' to exit, 'skip'/'s' to skip): ").strip().lower()
    if user_input in ['n', 'no', 'abort']:
        print("Aborting experiment sequence.")
        exit()
    elif user_input in ['skip', 's']:
        return False
    return True
