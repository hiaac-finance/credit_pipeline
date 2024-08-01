# %%

import subprocess
import itertools
import logging
import os
import time
from pathlib import Path

# Set up logging
logpath = Path('logs/runner_new_LC.log')
logpath.parent.mkdir(parents=True, exist_ok=True)

logging.getLogger().handlers = []

# Create and configure a custom logger for detailed (DEBUG level) logging
log = logging.getLogger('detailed')
log.setLevel(logging.DEBUG)  # Set this logger to capture everything

# Create a file handler for the custom logger (optional if you want all logs in the same file)
file_handler = logging.FileHandler(logpath)
file_handler.setLevel(logging.DEBUG)

# You might want to use the same formatter for consistency
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add the file handler to the detailed logger
log.addHandler(file_handler)



# %%
def run_command(cmd):
    """
    Executes a given command using subprocess and prints the output.
    """
    log.debug(f"Executing command: {' '.join(cmd)}")
    print(f"Executing command: {' '.join(cmd)}")
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout)
    if process.stdout:
        log.debug(f"Output:, {process.stdout}")
    if process.stderr:
        log.debug(f"Error:, {process.stderr}")
# %%

    # years = [2009]
    # seeds = [120054, 388388, 570334, 907360, 938870]
    # percent_bads = [0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.36, 0.4]
    # sizes = [1000, 5000, 10000]
    # contaminations = [0.12,]

def main():
    # Define argument values to iterate over
    ar_ranges = [(0, 100)]
    weights = [(1, 1)]
    years = [2009]
    seeds = [1]
    percent_bads = [0.06]
    sizes = [1000]
    contaminations = [0.12,]
    # For boolean flags, use a tuple with the argument name and a boolean to indicate if it should be included
    use_test_flags = [('--use_test', False)]
    train_ri_flags = [('--train_ri', True)] 
    reuse_exec_flags = [('--reuse_exec', False)]
    train_tn_flags = [('--train_tn', True)]
    eval_ri_flags = [('--eval_ri', True)]


    experiments = itertools.product(
        ar_ranges, weights, seeds, years, sizes, percent_bads, contaminations,
         use_test_flags, train_ri_flags, reuse_exec_flags, train_tn_flags, eval_ri_flags
    )
    qtd_exp = len(list(experiments))
    exp_n = 0

    log.debug(f'Running {qtd_exp} experiments.')
    # Iterate over all combinations of arguments
    for ar_range, weight, seed, year, size, percent_bad, contamination, use_test, train_ri, reuse_exec, train_tn, eval_ri in itertools.product(
        ar_ranges, weights, seeds, years, sizes, percent_bads, contaminations,
         use_test_flags, train_ri_flags, reuse_exec_flags, train_tn_flags, eval_ri_flags
    ):
        seed = seed + (year - 2009)
        cmd = [
            'python3', 'meta_LC exp.py',
            '--ar_range', str(ar_range[0]), str(ar_range[1]),
            '--weights', str(weight[0]), str(weight[1]),
            '--seed', str(seed),
            '--year', str(year),
            '--percent_bad', str(percent_bad),
            '--size', str(size),
            '--contamination', str(contamination)
        ]
        
        # Add flags if they are true
        if use_test[1]:
            cmd.append(use_test[0])
        if train_ri[1]:
            cmd.append(train_ri[0])
        if reuse_exec[1]:
            cmd.append(reuse_exec[0])
        if train_tn[1]:
            cmd.append(train_tn[0])
        if eval_ri[1]:
            cmd.append(eval_ri[0])

        exp_n += 1
        log.debug(f'Running experiment {exp_n} of {qtd_exp}')
        
        # Execute the constructed command
        run_command(cmd)
# %%
if __name__ == '__main__':
    log.debug('Starting runner in date: {0}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    main()

# %%