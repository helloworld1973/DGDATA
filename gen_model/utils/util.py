import random
import numpy as np
import torch


def log_and_print(content, filename):
    """
    Prints the content to the console and also writes it to a file.

    Args:
    - content (str): The content to be printed and logged.
    - filename (str): The name of the file to which the content will be written.
    """
    with open(filename, 'a') as file:  # 'a' ensures content is appended and doesn't overwrite existing content
        file.write(content + '\n')
    # print(content)


def print_row(row, colwidth=10, latex=False, file_name=''):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)
    Content = sep.join([format_val(x) for x in row])
    log_and_print(content=Content, filename=file_name)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def matrix_to_string(matrix):
    return '\n'.join(['\t'.join(map(str, row)) for row in matrix])
