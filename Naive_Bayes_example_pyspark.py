
"""Basic Example of using 
Naive Bayes Classification using pyspark

Usage : spark-submit Naive_Bayes_example_pyspark.py data/iris.data

Author: 
Abhineet Verma

"""

## Imports

from pyspark import SparkConf, SparkContext

from operator import add
import sys
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


## Constants
APP_NAME = "Naive Bayes Example"
##OTHER FUNCTIONS/CLASSES

def main(spark,filename):
  df = spark.read.csv(filename,header=False,inferSchema=True)
  vector_assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'],outputCol='features')
# >>> df.show(4)
# +---+---+---+---+-----------+
# |_c0|_c1|_c2|_c3|        _c4|
# +---+---+---+---+-----------+
# |5.1|3.5|1.4|0.2|Iris-setosa|
# |4.9|3.0|1.4|0.2|Iris-setosa|
# |4.7|3.2|1.3|0.2|Iris-setosa|
# |4.6|3.1|1.5|0.2|Iris-setosa|
# +---+---+---+---+-----------+
  vector_assembler = VectorAssembler(inputCols=['_c0','_c1','_c2','_c3'],outputCol='features')
  v_df = vector_assembler.transform(df)

# >>> v_df.show(4)
# +---+---+---+---+-----------+-----------------+
# |_c0|_c1|_c2|_c3|        _c4|         features|
# +---+---+---+---+-----------+-----------------+
# |5.1|3.5|1.4|0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|
# |4.9|3.0|1.4|0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|
# |4.7|3.2|1.3|0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|
# |4.6|3.1|1.5|0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|
# +---+---+---+---+-----------+-----------------+
# only showing top 4 rows
  indexer = StringIndexer(inputCol='_c4',outputCol='label')
  i_df = indexer.fit(v_df).transform(v_df)
#   >>> i_df.show(4)
# +---+---+---+---+-----------+-----------------+-----+
# |_c0|_c1|_c2|_c3|        _c4|         features|label|
# +---+---+---+---+-----------+-----------------+-----+
# |5.1|3.5|1.4|0.2|Iris-setosa|[5.1,3.5,1.4,0.2]|  0.0|
# |4.9|3.0|1.4|0.2|Iris-setosa|[4.9,3.0,1.4,0.2]|  0.0|
# |4.7|3.2|1.3|0.2|Iris-setosa|[4.7,3.2,1.3,0.2]|  0.0|
# |4.6|3.1|1.5|0.2|Iris-setosa|[4.6,3.1,1.5,0.2]|  0.0|
# +---+---+---+---+-----------+-----------------+-----+
# only showing top 4 rows
  splits = i_df.randomSplit([0.6,0.4],1)
  train_df =  splits[0]
  test_df = splits[1]
  nb = NaiveBayes(modelType='multinomial')
  nbmodel  = nb.fit(train_df)
  predictions = nbmodel.transform(test_df)
  evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
  nbaccuracy = evaluator.evaluate(predictions)
  print(nbaccuracy)
 

if __name__ == "__main__":

   # Configure Spark
   # conf = SparkConf().setAppName(APP_NAME)
   # conf = conf.setMaster("local[*]")
   # sc   = SparkContext(conf=conf)
   filename = sys.argv[1]
   spark = SparkSession\
        .builder\
        .appName(APP_NAME)\
        .getOrCreate()
   # Execute Main functionality
   main(spark, filename)
