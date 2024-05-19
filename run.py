#import sys
#sys.path.append('/home/guest/公共/wsf/ann_benchmarks/ann_benchmarks/runner.py')
from multiprocessing import freeze_support

from ann_benchmarks.main import main

if __name__ == "__main__":
    freeze_support()
    main()
