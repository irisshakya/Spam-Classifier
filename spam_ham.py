
import pandas as pd
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer, RegexTokenizer, NGram, HashingTF, IDF
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql import SparkSession

# spark setup
from pyspark import SparkContext, SparkConf
sc = SparkContext(master = 'local')
spark = SparkSession.builder.appName('pySpark word-count').config('spark.some.config.option', 'some-value')             .getOrCreate()


# data import
data = spark.read.format("csv").option("header",True).load("/Users/iris/spam_data.csv") \
    .withColumnRenamed("Category", "label") \
    .withColumnRenamed("Message", "text")
data_df = data.toDF("label", "text")


data.groupBy('label').count().show()

data_df.show(3)
print(data_df.count(), data_df.columns)
type(data_df)


from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
# replace label spam = 0 and ham = 1
data_df = data_df.withColumn('label', regexp_replace('label', 'spam', '0'))
data_df = data_df.withColumn('label', regexp_replace('label', 'ham', '1'))

# cast those string categories as int dtype
data_df = data_df.withColumn('label', data_df.label.cast(IntegerType()))

print(data_df.printSchema())


# # pre-processing
# post_data = Tokenizer(inputCol='text', outputCol='words').transform(data_df)

# #test_post_data = RegexTokenizer(inputCol='text', outputCol='token', pattern='\\W+')

# #countTokens = udf(lambda x: len(x), IntegerType())
# post_data.show(4)
# post_data.columns
# from pyspark.sql.functions import col 
# post_data.select('label', 'text', 'words').withColumn('token', countTokens(col('words'))).show(5)

data_df.show(4)

data_df = data_df.dropna()
data_df.groupBy('label').count().show()


# 1. tokenize and clean data sentences using Tokenizer
tokenizer = Tokenizer(inputCol='text', outputCol='words')
tokenized_df = tokenizer.transform(data_df)

# 2. BiGram the words
bigrams = NGram(n=2, inputCol='words', outputCol='bigrams')
bigramed_df = bigrams.transform(tokenized_df)

# 3. hashTF the data
hasherTF = HashingTF(inputCol='bigrams', outputCol='hasher', numFeatures=6000)
hashed_df = hasherTF.transform(bigramed_df)

# 4. IDF
idf = IDF(inputCol='hasher', outputCol='features')
idf_df = idf.fit(hashed_df)
output_df = idf_df.transform(hashed_df)


output_df = output_df.select('features', 'label')
output_df.show(5)

train_1, test_1 = output_df.randomSplit([0.8, 0.2], seed=666)
train_2, test_2 = output_df.randomSplit([0.8, 0.2], seed=768)
train_3, test_3 = output_df.randomSplit([0.8, 0.2], seed=563)
train_4, test_4 = output_df.randomSplit([0.8, 0.2], seed=356)
train_5, test_5 = output_df.randomSplit([0.8, 0.2], seed=879)



# train test
dt_clf_1 = DecisionTreeClassifier(featuresCol='features', labelCol="label").fit(train_1)
dt_pred_1 = dt_clf_1.transform(test_1)
dt_pred_1.show(5)


check_accuracy = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='accuracy') \
                .evaluate(dt_pred_1)
check_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')   \
                .evaluate(dt_pred_1)
check_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall') \
                .evaluate(dt_pred_1)
check_f = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='f1')  \
                .evaluate(dt_pred_1)


print('accuracy: %0.4f' % check_accuracy)
print('precision: %0.4f' % check_precision)
print('recall: %0.4f' % check_recall)
print('F1 score: %0.4f' % check_f)


dt_clf_1.featureImportances


# # from pyspark.ml.classification import RandomForestClassifier
# # rf = RandomForestClassifier(labelCol='label', featuresCol='')
# eval = BinaryClassificationEvaluator()

# training_accuracy = eval.evaluate(training_predictions)
# # print('Training set accuracy: {:.4g}.'.format(training_accuracy))
# test_accuracy = eval.evaluate(test_predictions)
# print('Test accuracy: {:.4g}'.format(test_accuracy))


# create Cross-Validator and ParamGrid
# pipeline = Pipeline(stages = [tokenizer, bigrams, hasherTF, idf])
params = ParamGridBuilder().build()

# Run k-fold Cross-Validator 
cv = CrossValidator(estimator=DecisionTreeClassifier(),\
     estimatorParamMaps=params,\
     evaluator=MulticlassClassificationEvaluator(), \
     seed=645,numFolds=5)
cv_model_1 = cv.fit(train_1) 
cv_model_2 = cv.fit(train_2) 
cv_model_3 = cv.fit(train_3) 
cv_model_4 = cv.fit(train_4) 
cv_model_5 = cv.fit(train_5) 

print('CV_1: %0.4f' %cv_model_1.avgMetrics[0])
print('CV_2: %0.4f' %cv_model_2.avgMetrics[0])
print('CV_3: %0.4f' %cv_model_3.avgMetrics[0])
print('CV_4: %0.4f' %cv_model_4.avgMetrics[0])
print('CV_5: %0.4f' %cv_model_5.avgMetrics[0])


# prediction model
cv_prediction_1 = cv_model_1.transform(test_1)
cv_prediction_1 = cv_prediction_1.select('label', 'prediction')

cv_prediction_2 = cv_model_2.transform(test_2)
cv_prediction_2 = cv_prediction_2.select('label', 'prediction')

cv_prediction_3 = cv_model_3.transform(test_3)
cv_prediction_3 = cv_prediction_3.select('label', 'prediction')

cv_prediction_4 = cv_model_3.transform(test_4)
cv_prediction_4 = cv_prediction_4.select('label', 'prediction')

cv_prediction_5 = cv_model_4.transform(test_5)
cv_prediction_5 = cv_prediction_5.select('label', 'prediction')



cv_prediction_1.groupBy('label','prediction').count().show()
pred_1_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')
pred_1_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall')
precision_1 = pred_1_precision.evaluate(cv_prediction_1)
recall_1 = pred_1_recall.evaluate(cv_prediction_1)

cv_prediction_2.groupBy('label','prediction').count().show()
pred_2_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')
pred_2_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall')
precision_2 = pred_2_precision.evaluate(cv_prediction_2)
recall_2 = pred_2_recall.evaluate(cv_prediction_2)

pred_3_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')
pred_3_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall')
precision_3 = pred_3_precision.evaluate(cv_prediction_3)
recall_3 = pred_3_recall.evaluate(cv_prediction_3)

pred_4_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')
pred_4_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall')
precision_4 = pred_2_precision.evaluate(cv_prediction_4)
recall_4 = pred_4_recall.evaluate(cv_prediction_4)

pred_5_precision = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedPrecision')
pred_5_recall = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label', metricName='weightedRecall')
precision_5 = pred_5_precision.evaluate(cv_prediction_5)
recall_5 = pred_5_recall.evaluate(cv_prediction_5)

print('Precision_1: %0.4f' %precision_1)
print('Recall_1: %0.4f \n' %recall_1)

print('Precision_2: %0.4f' %precision_2)
print('Recall_1: %0.4f\n' %recall_2)

print('Precision_3: %0.4f' %precision_3)
print('Recall_1: %0.4f\n' %recall_3)

print('Precision_4: %0.4f' %precision_4)
print('Recall_1: %0.4f\n' %recall_4)

print('Precision_5: %0.4f' %precision_5)
print('Recall_1: %0.4f\n' %recall_5)

# mean results for precision and recall for each split data
from statistics import mean
nums_precision = [precision_1, precision_2, precision_3, precision_4, precision_5]
avgP = mean(nums_precision)
print('The average for Precision is: %0.4f' % avgP)

nums_recall = [recall_1, recall_2, recall_3, recall_4, recall_5]
avgR = mean(nums_recall)
print('The average for Recall is: %0.4f' %avgR)