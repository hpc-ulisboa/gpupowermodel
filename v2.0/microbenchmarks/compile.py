#!/usr/bin/python2
import sys
import subprocess
import os
import time

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

arch = '52'

print '\nCompiling microbenchmarks for arch=sm_{}.'.format(arch)

microbenchmarks = ['DRAM', 'L2_Cache', 'SF', 'Shared_Mem', 'FP32/FP32_ADD', 'FP32/FP32_MUL', 'FP32/FP32_FMA', 'FP64/FP64_ADD', 'FP64/FP64_MUL', 'FP64/FP64_FMA', 'CF']

for bench_id,bench in enumerate(microbenchmarks):
    print '\n\n' + bench+':'
    with cd('{}'.format(bench)):
        command= 'make clean'
        print '\n' + command
        subprocess.call(command, shell=True)
        command= 'make ARCH={}'.format(arch)
        print '\n' + command
        subprocess.call(command, shell=True)

print '\n\nFinished compiling microbenchmarks'
