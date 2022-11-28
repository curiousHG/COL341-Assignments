import sys
import os
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

print("Enter you name:")
name = input().strip()
os.makedirs(name, exist_ok=True)
name+="/"
path = 'data/train_large.csv'     ##seed_file
path2 = "data/train.csv"         ## testing seed file
train_name = name + '_train_'
test_name = name + '_test_'
output_name = name+'_out_'
correct_name = name+'_correct_'

# files have increasing number of rows in the ratio 1:2:3:4:5... :numfiles
print(" Generating training_sets ")
numFiles = 5

Trainsizes=np.zeros(numFiles+1)
Testsizes=np.zeros(numFiles+1)

m1,m2=0,0
for i in range (1,numFiles+1):
    df = pd.read_csv(path)
    m1 = df.shape[0]
    df= df.iloc[0: (i*m1)//numFiles ]
    Trainsizes[i]= (i*m1)//numFiles 
    df.to_csv(train_name+str(i)+".csv",index=False)
    del df
    print("Creation Trainingset "+ str(i)+" done")

print("Generating test_sets now")

for i in range(1,numFiles+1):
    df = pd.read_csv(path2)
    m2 = df.shape[0]
    df= df.iloc[0: (i*m2)//numFiles ]
    Testsizes[i]= (i*m2)//numFiles 
    y= df["Total Costs"].to_numpy()
    df.drop(df.columns[[-1]], axis = 1, inplace = True)
    df.to_csv(test_name+str(i)+".csv",index=False)
    np.savetxt(correct_name+str(i) , y, delimiter="\n")
    print("Creating Testingset"+ str(i)+" done")
    del df
print("Testcases generation done")

R2=[]

print("Beginning the Training and Testing Iterations")

for i in range(1,numFiles+1):
    print("Training your model iteration = "+str(i)+" with trainsize = ",Trainsizes[i],"\n")
    for j in range(1,numFiles+1):
        print("Testing on ", str(Testsizes[j])," datasize" )
        process = Popen(['python3','linear.py','c', train_name + str(i) + '.csv', test_name + str(j) + '.csv',output_name + str(i)+"_"+str(j) + '.txt'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        y = np.loadtxt(correct_name+str(j),dtype = float)
        out_y = np.loadtxt(output_name +str(i)+"_"+ str (j) + '.txt', dtype=float)
        r2= r2_score( y, out_y)
        R2.append(r2)
        print('r2_score at testing iteration = {}: <{}>'.format (j,r2))
    print("")
print("Average r2score over", numFiles* numFiles," iterations = ", sum(R2)/len(R2) )