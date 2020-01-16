#!/bin/bash

original="data/"
replace="../input/ieee-fraud-detection/"

sed -i s@$original@$replace@g src/main.py
kaggle kernels push
sed -i s@$replace@$original@g src/main.py
