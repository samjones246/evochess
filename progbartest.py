from tqdm import tqdm
from time import sleep
from itertools import count

for i in tqdm(count(0), "Outer", unit=""):
    if i==5:
        break
    sleep(0.5)