from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
import itertools

test_labels=[]
data = sc.textFile("/FileStore/tables/glass.data")
data=data.map(lambda x: str(x).split(","))
glass_data, test = data.randomSplit([0.6, 0.4])
glass_data=glass_data.map(lambda row: LabeledPoint(row[-1], row[:-1]))
test=test.collect()
for i in range(len(test)):
  test_labels.append(float(test[i][10]))
  del test[i][10]  
test=sc.parallelize(test) 

# Train a naive Bayes model.
model = DecisionTree.trainClassifier(glass_data,8,{})

# Make prediction.
prediction = model.predict(test)
predictions=prediction.collect()
predictionAndLabel= zip(test_labels, predictions)
predictionAndLabel=sc.parallelize(predictionAndLabel)
accuracy = 100 * float(predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count()) / test.count()
print('Model Accuracy: {}'.format(accuracy))
