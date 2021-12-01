# -*- coding: utf-8 -*-

"""
BIG DATA PRACTICAL APPLICATION
Itziar Alonso, Miriam Gutierrez, Elia Alonso
"""
#%%
"LOAD THE INPUT DATA"
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum
import pyspark.sql.functions as F
from pyspark.sql.functions import mean as _mean
from pyspark.ml.feature import (OneHotEncoder, StringIndexer)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import MinMaxScaler



spark = SparkSession.builder.appName('ml-1987').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
 
df = spark.read.csv('C:/Users/Elia/Desktop/master/BIG DATA/PracticalApplication/1987.csv', header = True, inferSchema = True)


#%%
"SHOW THE NUMBER OF INSTANCES AND COLUMNS OF THE INPUT DATA SET"
print("Number of instances of the original dataset: ",df.count())
print("Number of columns of the original dataset", len(df.columns))
print("Type of each variable:")
df.printSchema()
#Info extra que meto aqui por si nos sirve en algun momento
#Para ver los distintos valores que puede tener una columna: 
#df.select('COLUMNA').distinct().show()
#%%
"PRE-PROCESSING"

"0- Turn integer into string types to convert them in categorical variables: Year, Month, DayofMonth, DayOfWeek"
change_type = ["Month", "DayofMonth", "DayOfWeek"]
#Year should be in change_type but as it has only 1 value (1987), it fails in the one-hot encoding step; if there would be more than one value, it would work
print("The numerical variables being converted into categorical are:")
df_cat=df
for col in range(0, len(df_cat.columns)):
    name_col=df_cat.columns[col]
    if name_col in change_type:
        print(name_col)
        df_cat = df_cat.withColumn(name_col, df_cat[name_col].cast(StringType()))
print("Therefore, the type of each variable now is:")
df_cat.printSchema()       

#%%
"1- Remove forbidden variables"
print("Removing forbidden variables...")
df_drop = df_cat.drop(*['ArrTime','ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'])
df_drop.columns
print("Schema of the data set after removing the forbidden variables:")
df_drop.printSchema()
print("Number of columns after removing the forbidden variables: ", len(df_drop.columns))
#%%
"2- Replace NA por null for Spark to recognize them"
print("Replacing missing values for null...")
df_rep = df_drop.replace(("NA"), None)
#%%
"3- Remove cancelled flights"
print("Deleting cancelled flights...")
df_not_can=df_rep.filter(df_rep.Cancelled == 0)
"Remove Cancelled column since all the cancelled flights have been removed"
df_not_can = df_not_can.drop(*['Cancelled'])
print("Number of non cancelled flights: ", df_not_can.count())
#%%
"4- Count the missing values:"
def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    df = spark_df.select([F.count(F.when(F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes ]).toPandas()
    if len(df) == 0:
        print("There are no any missing values!")
        return None
    if sort:
        return df.rename(index={0: 'count'}).T.sort_values("count",ascending=False)
    return df

n_missings=count_missings(df_not_can,sort=True)
print("Number of missing values in each attribute:", n_missings)
#%%
"5- If the number of missing values of each variable is over the 70% of the total number of instances, then the variable is removed from the dataframe"
print("Removing the variables with missing values over the 70% of the total number of instances...")
list_missing=count_missings(df_not_can,sort=False)
list_missing_array=list_missing.values.tolist()
list_missing_array=list_missing_array[0]
list_drop=[]
print("Columns dropped from the dataframe:")
df_before_dropping = df_not_can
for col in range(0, len(df_before_dropping.columns)):
    if list_missing_array[col]>0.7*df_before_dropping.count():
        list_drop.append(df_before_dropping.columns[col])
for i in range(0, len(list_drop)):
    print(list_drop[i])
    df_before_dropping = df_before_dropping.drop(list_drop[i])
df_dropped_missing=df_before_dropping
print("Total: ",len(df_not_can.columns)-len(df_dropped_missing.columns), 'columns dropped')
#%%
"6- Fill null values in the variables that have not been removed"
print("Missing values from the rest of the columns with less than a 70% are filled as follows:")
print("- Categorical values: with 0")
print("- Numerical values: with the mean of the column")
print("")
catCols = [x for (x, dataType) in df_dropped_missing.dtypes if dataType =="string"]
numCols = [ x for (x, dataType) in df_dropped_missing.dtypes if dataType !="string"]
print("Categorical variables: ", catCols)
print("Numerical variables: ", numCols)
for col in range(0, len(df_dropped_missing.columns)):
    name_col=df_dropped_missing.columns[col]
    if name_col in catCols: # If categorical
        df_dropped_missing=df_dropped_missing.fillna({ name_col:0} )
    else:                                           #if numerical
        df_stats = df_dropped_missing.select( _mean(name_col).alias('mean')).collect()
        mean = df_stats[0]['mean']
        df_dropped_missing=df_dropped_missing.fillna(mean, subset=[name_col])
clean_df=df_dropped_missing
"7- Checking that the missing values have been properly removed or filled"
print("Checking that the missing values have been properly removed or filled...")
print("Number of missing values in each attribute: ", count_missings(clean_df, sort=True))
#%%
"8- One-hot encoding for categorical variables"
print("Transforming categorical variables using one-hot encoding...")
print("Variables to be transformed:", catCols)

string_indexer = [
    StringIndexer(inputCol=x, outputCol=x + "_StringIndexer", handleInvalid="skip")
    for x in catCols
]

one_hot_encoder = [
    OneHotEncoder(
        inputCols=[f"{x}_StringIndexer" for x in catCols],
        outputCols=[f"{x}_OneHotEncoder" for x in catCols],
    )
]

# assemblerInput = [x for x in numCols]

# assemblerInput += [f"{x}_OneHotEncoder" for x in catCols]
# assemblerInput
# vector_assembler = VectorAssembler(
#     inputCols=assemblerInput, outputCol="VectorAssembler_features"
# )

stages=[]
stages += string_indexer
stages += one_hot_encoder
#stages += [vector_assembler]

pipeline = Pipeline().setStages(stages)
model = pipeline.fit(clean_df)
df_encoded = model.transform(clean_df)
print("Let's see how the data set looks like after one-hot encoding categorical variables:")
df_encoded.show(10)
df_encoded.printSchema()

notStringCols = [x for (x, dataType) in df_encoded.dtypes if ((dataType !="string") and (dataType !="double"))]
df_encoded_clean = df_encoded.select([col for col in notStringCols])
df_encoded_clean.printSchema()
#%%

#%%
"MACHINE LEARNING MODEL"

"Splitting the data set in training and test subsets"

train, test = df_encoded_clean.randomSplit([0.7, 0.3], seed=7)
print("Splitting the data set in training and test subsets with 70% and 30% respectively...")
print("Train set length: ", train.count())
print("Test set length: ", test.count())
#%%
integer_cols = [x for (x, dataType) in train.dtypes if (dataType =="int")]
#%%
for i in integer_cols:
    print(i)
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    
    train_scaled = pipeline.fit(train).transform(train).withColumn(i+"_Scaled", col(i+"_Scaled")).drop(i+"_Vect")
                                                                   

print("After Scaling :")
train_scaled.show(10)

# df.withColumn("salary",col("salary")*100).show()

#%%
"MACHINE LEARNING VALIDATION"
