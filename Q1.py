from pyspark.mllib.clustering import KMeans, KMeansModel

values=[]
data = sc.textFile("/FileStore/tables/itemusermat")
movie_data = sc.textFile("/FileStore/tables/movies.dat")
data=data.map(lambda x: str(x).split(" ")).map(lambda row:(int(row[0]), row[1:]))
ratings=data.map(lambda x: x[1])
movie_data=movie_data.map(lambda x: x.split("::")).map(lambda row: (int(row[0]), row[1:]))
joined_data=movie_data.join(data)
model = KMeans.train(ratings, 10,maxIterations=10, initializationMode="random")
res=data.map(lambda dat: (model.predict(dat[1]),dat[0])).map(lambda x: (x[1],x[0]))
result=res.join(joined_data).map(lambda x: (int(x[1][0]),x[0],x[1][1][0])).collect()
cluster_num=5
for i in result:
  if i[0]== cluster_num:
    values.append(i)
if len(values)>=5: 
  for i in range(0,5):
    print(values[i])
else:
  for i in range(0,len(values)):
    print(values[i])