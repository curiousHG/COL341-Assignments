import os
import re
import sys
import numpy as np

path = sys.argv[1]
files = os.listdir(path)
r1=re.compile("w_.\.npy")#use w_._iter\.npy for weights after iteration
r2=re.compile("ac_w_.\.npy")#use ac_w_._iter\.npy for weights after iteration
obtained_weights = list(filter(r1.match,files))
actual_weights = list(filter(r2.match,files))
print(obtained_weights) #check the files read
print(actual_weights) #check the files read
if not obtained_weights or not actual_weights:
    print("One or both names do not exist")
    exit()

if(len(obtained_weights)!=len(actual_weights)):
        print("Weight files are less or more")
        exit()

for i in range(len(actual_weights)):
    ac_w = np.load(path + '\\'+ actual_weights[i])
    ob_w = np.load(path + '\\'+ obtained_weights[i])
    if(ob_w.shape!=ac_w.shape):
        print('Either files are in wrong order or shape problem with file w_'+str(i)+'.npy')
        exit()

scale = 0
flag=0

for i in range(len(actual_weights)):
    ac_w = np.load(path+ '\\'+ actual_weights[i])
    ob_w = np.load(path + '\\'+ obtained_weights[i])
    error = np.abs(ac_w-ob_w)
    if np.any(error>1e-15):
        flag=1
        print("Weight mismatch in " + str(obtained_weights[i]) + " at following positions, row coordinates followed by column coordinates")
        print(np.where(error>1e-15))

if flag==0:
    print("Weights match")
