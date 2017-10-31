curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/Index	 -o Index.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/WARNING	 -o WARNING.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ask-detrano	 -o ask-detrano.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/bak		  -o bak.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleve.mod -o cleve.mod.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/cleveland.data -o cleveland.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names  -o heart-disease.names.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/hungarian.data -o hungarian.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/long-beach-va.data -o long-beach-va.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/new.data	  -o new.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data  -o processed.cleveland.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data -o processed.hungarian.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data  -o processed.switzerland.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data	  -o processed.va.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data -o reprocessed.hungarian.data.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/switzerland.data  -o switzerland.data.txt

mkdir costs

curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/costs/heart-disease.README -o costs/heart-disease.README.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/costs/heart-disease.cost -o costs/heart-disease.cost.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/costs/heart-disease.delay -o costs/heart-disease.delay.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/costs/heart-disease.expense -o costs/heart-disease.expense.txt
curl http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/costs/heart-disease.group -o costs/heart-disease.group.txt