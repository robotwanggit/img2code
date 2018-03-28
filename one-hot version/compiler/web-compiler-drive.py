#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys, os

from os.path import basename
from classes.Utils import *
from classes.Compiler import *

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[-3:] == 'gui':
        input_files = [argv[0]]
    else:
        input_files = [argv[0]+file_name for file_name in os.listdir(argv[0]) if file_name[-3:] == 'gui']

FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

dsl_path = "./drive/AI/img2code/compiler/assets/web-dsl-mapping.json"
compiler = Compiler(dsl_path)

for input_file in input_files:
    file_uid = basename(input_file)[:basename(input_file).find(".")]
    path = input_file[:input_file.find(file_uid)]
    input_file_path = "{}{}.gui".format(path, file_uid)
    output_file_path = "{}{}.html".format(path, file_uid)
    print(output_file_path)
    with open(input_file_path, 'r') as f:
        tokens = f.read()
    result = compiler.compile(tokens, output_file_path)