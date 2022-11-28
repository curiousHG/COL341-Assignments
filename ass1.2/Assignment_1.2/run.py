import subprocess
case = input("Enter the case number: ")
type = input("Enter the type: ")


# python3 grade_a.py outputs/1/outputfile.txt outputs/1/weightfile.txt testcases/1/outputfile1.txt testcases/1/weightfile1.txt
subprocess.call(["mkdir", "-p", f'outputs/{case}'])
subprocess.call(
    [
        "time","python3", "logistic.py", type, 
        "data/train.csv", "data/test.csv", 
        f'testcases/{case}/param{case}.txt', 
        f'outputs/{case}/outputfile.txt', f'outputs/{case}/weightfile.txt'
    ]
)
subprocess.call(
    [
        "python3", f'grade_{type}.py', f'outputs/{case}/outputfile.txt', f'outputs/{case}/weightfile.txt', 
        f'testcases/{case}/outputfile{case}.txt', f'testcases/{case}/weightfile{case}.txt'])