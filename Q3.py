from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import itertools

test_ratings=[]
ratings = sc.textFile("/FileStore/tables/ratings.dat")
ratings=ratings.map(lambda x: x.split("::")).map(lambda row: [int(row[0]), int(row[1]),float(row[2])])
training, test = ratings.randomSplit([0.6, 0.4])
test_data=test.map(lambda r: (r[0],r[1]))
training=training.map(lambda row: Rating(row[0],row[1],row[2]))

rank = 50
numIterations =20
model = ALS.train(training, rank, numIterations,0.01)
predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
pred=ratesAndPreds.map(lambda r:(r[1][0],round(r[1][1])))
accuracy = 100 *(pred.filter(lambda pl: pl[0] == pl[1]).count())/ test.count()
print('Model Accuracy: {}'.format(accuracy))