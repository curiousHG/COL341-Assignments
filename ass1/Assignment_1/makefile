a: 
	time python3 linear.py a data/train.csv data/test.csv WO/out_a.txt WO/w_a.txt
	python3 grade_a.py WO/out_a.txt WO/w_a.txt WO/model_outputfile_a.txt WO/model_weightfile_a.txt
b:
	time python3 linear.py b data/train.csv data/test.csv WO/regularization.txt WO/out_b.txt WO/w_b.txt WO/best_lamba.txt
	python3 grade_b.py WO/out_b.txt WO/w_b.txt WO/model_outputfile_b.txt WO/model_weightfile_b.txt
c:
	time python3 linear.py c data/train_large.csv data/test.csv WO/out_c.txt
	python3 grade_b.py WO/out_c.txt WO/model_weightfile_b.txt WO/xgboost_r2-85.txt WO/model_weightfile_b.txt
	python3 grade_c.py

clean:
	rm -f WO/out_a.txt WO/w_a.txt WO/out_b.txt WO/w_b.txt WO/out_c.txt WO/w_c.txt WO/best_lamba.txt 