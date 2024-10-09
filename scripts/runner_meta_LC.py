# %%

import subprocess
import multiprocessing
import itertools
import logging
from loguru import logger
import os
import time
from pathlib import Path

# Set up logging
logpath = Path('logs/runner_new_LC.log')
logpath.parent.mkdir(parents=True, exist_ok=True)

# #new logger
# logger.remove()
# # logger.add(logpath, level="TRACE", rotation="100 MB")
# logger.add(logpath, format="{time} - {line} - {message}", level="TRACE", rotation="100 MB", enqueue=True)

# Função para limpar manipuladores existentes antes de adicionar novos
def setup_custom_logger(logpath):
    log = logging.getLogger('detailed')
    
    # Limpar manipuladores antigos para evitar logs duplicados
    if log.hasHandlers():
        log.handlers.clear()
    
    log.setLevel(logging.INFO)  # Definir o nível do logger
    
    # Criar um manipulador de arquivo
    file_handler = logging.FileHandler(logpath)
    file_handler.setLevel(logging.INFO)
    
    # Definir o formato do logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Adicionar o manipulador ao logger
    log.addHandler(file_handler)
    
    return log
# ----------------------------------------------------------------------------

#old logger
logging.getLogger().handlers = []

# Create and configure a custom logger for detailed (DEBUG level) logging
log = logging.getLogger('detailed')
log.setLevel(logging.INFO)  # Set this logger to capture everything

# Create a file handler for the custom logger (optional if you want all logs in the same file)
file_handler = logging.FileHandler(logpath)
file_handler.setLevel(logging.INFO)

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
    logger.info(f"Executing command: {' '.join(cmd)}")
    print(f"Executing command: {' '.join(cmd)}")
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(process.stdout)
    # print(f"Error: {process.stderr}")
    if process.stdout:
        logger.info(f"Output:, {process.stdout}")
    if process.stderr:
        logger.info(f"Error:, {process.stderr}")
# %%

    # years = [2009]
    # seeds = [120054, 388388, 570334, 907360, 938870] [120054, 388388, 570337, 907361, 938870]
    # percent_bads = [0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.36, 0.4]
    # sizes = [1000, 5000, 10000]
    # contaminations = [0.12,]

def main():
    # Define argument values to iterate over
    ar_ranges = [(0, 100)]
    weights = [(1, 1)]
    years = [2014, 2015]#,2009, 2010, 2011, 2012, 2013, 2014, 2015]
    seeds = [120054, 388388, 570337, 907361, 938870, 555555,555556, 555557, 555558, 555559,
            800000, 800001, 800002, 800003, 800004, 800005, 800006, 800007, 800008, 800009,]
    # [120054, 388388, 570337, 907361, 938870, 555555,555556, 555557, 555558, 555559,
    #          800000, 800001, 800002, 800003, 800004, 800005, 800006, 800007, 800008, 800009,]
    # seeds = []
    # 388388, 570337, 907361, 938870, 555555, 555556, 555557, 555558, 555559,
    #          800000, 800001, 800002, 800003, 800004, 800005, 800006, 800007, 800008, 800009,
            #  ]
    percent_bads = [0.07]
    sizes = [1000]
    contaminations = [0.12,]
    # For boolean flags, use a tuple with the argument name and a boolean to indicate if it should be included
    use_test_flags = [('--use_test', True)]
    train_ri_flags = [('--train_ri', True)] 
    reuse_exec_flags = [('--reuse_exec', True)]
    train_tn_flags = [('--train_tn', True)]
    eval_ri_flags = [('--eval_ri', True)]


    experiments = itertools.product(
        ar_ranges, weights, seeds, years, sizes, percent_bads, contaminations,
         use_test_flags, train_ri_flags, reuse_exec_flags, train_tn_flags, eval_ri_flags
    )
    qtd_exp = len(list(experiments))
    exp_n = 0

    logger.info(f'Running {qtd_exp} experiments.')
    # Iterate over all combinations of arguments
    for ar_range, weight, seed, year, size, percent_bad, contamination, use_test, train_ri, reuse_exec, train_tn, eval_ri in itertools.product(
        ar_ranges, weights, seeds, years, sizes, percent_bads, contaminations,
         use_test_flags, train_ri_flags, reuse_exec_flags, train_tn_flags, eval_ri_flags
    ):
        seed = seed + (year - 2009)
        cmd = [
            'python3', 'meta_LC.py',
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
        logger.info(f'Running experiment {exp_n} of {qtd_exp}')
        
        # Execute the constructed command
        run_command(cmd)
# %%


def run_experiment(args):
    ar_range, weight, seed, year, size, percent_bad, contamination, use_test, train_ri, reuse_exec, train_tn, eval_ri = args
    
    # Ajustar a seed com base no ano
    seed = seed + (year - 2009)
    cmd = [
        'python3', 'meta_LC.py',
        '--ar_range', str(ar_range[0]), str(ar_range[1]),
        '--weights', str(weight[0]), str(weight[1]),
        '--seed', str(seed),
        '--year', str(year),
        '--percent_bad', str(percent_bad),
        '--size', str(size),
        '--contamination', str(contamination)
    ]
    
    # Adicionar flags se forem verdadeiras
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

    # Executar o comando
    run_command(cmd)

# def main2():
#     logger = logging.getLogger('detailed')
#     # Definir valores de argumentos para iteração
#     ar_ranges = [(0, 100)]
#     weights = [(1, 1)]
#     years = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
#     seeds = [120054, 388388, 570337, 907361, 938870, 555555, 555556, 555557, 555558, 555559,
#              800000, 800001, 800002, 800003, 800004, 800005, 800006, 800007, 800008, 800009]
#     percent_bads = [0.07]
#     sizes = [1000]
#     contaminations = [0.12]
#     use_test_flags = [('--use_test', True)]
#     train_ri_flags = [('--train_ri', True)]
#     reuse_exec_flags = [('--reuse_exec', True)]
#     train_tn_flags = [('--train_tn', True)]
#     eval_ri_flags = [('--eval_ri', True)]
    
#     # Criar combinações de experimentos
#     experiments = itertools.product(
#         ar_ranges, weights, seeds, years, sizes, percent_bads, contaminations,
#         use_test_flags, train_ri_flags, reuse_exec_flags, train_tn_flags, eval_ri_flags
#     )
    
#     experiments_list = list(experiments)  # Converter para lista para contar o número total
#     qtd_exp = len(experiments_list)
#     logger.info(f'Running {qtd_exp} experiments.')
    
#     #Usar multiprocessing para rodar os experimentos em paralelo
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         pool.map(run_experiment, experiments_list)


if __name__ == '__main__':
    setup_custom_logger(logpath) 
    logger = logging.getLogger('detailed')
    logger.info(" " * 50)
    logger.info("-" * 50)
    logger.info('Starting runner in date: {0}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    main()

# %%