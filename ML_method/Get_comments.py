import os
import pandas as pd
import glob

csv_list = glob.glob('../data/*.csv')

for i in csv_list:
    print(i)
    fr = open(i, 'rb').read()
    with open ('comments.csv', 'ab') as f:
        f.write(fr)