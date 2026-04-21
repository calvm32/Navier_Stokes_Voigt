import time
from pathlib import Path

from processing.printoff import iter_info_verbose, text, green
from processing.statistics.pdf_sampler import pdf_sampler
from processing.statistics.structure_funcs import structure_funcs
from processing.post_processing import *