# -*- coding: utf-8 -*-

"""
BIG DATA PRACTICAL APPLICATION
Itziar Alonso, Miriam Gutierrez, Elia Alonso
"""
#%%
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum
import pyspark.sql.functions as F
from pyspark.sql.functions import mean as _mean
from pyspark.ml.feature import (OneHotEncoder, StringIndexer)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType



spark = SparkSession.builder.appName('ml-1987').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

"LOAD THE INPUT DATA"
print("Loading input data...")

df = spark.read.csv('C:/Users/Elia/Desktop/master/BIG DATA/PracticalApplication/1987.csv', header = True, inferSchema = True)


#%%
"SHOW THE NUMBER OF INSTANCES AND COLUMNS OF THE INPUT DATA SET"
print("Number of instances of the original dataset: ",df.count())
print("Number of columns of the original dataset", len(df.columns))
print("Type of each variable:")
df.printSchema()
#Info extra que meto aqui por si nos sirve en algun momento
#Para ver los distintos valores que puede tener una columna: 
#%%
"PRE-PROCESSING"
#%%
"0- Remove forbidden variables"
print("Removing forbidden variables...")
df_drop_forbidden = df.drop(*['ArrTime','ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'])
df_drop_forbidden.columns
print("Schema of the data set after removing the forbidden variables:")
df_drop_forbidden.printSchema()
print("Number of columns after removing the forbidden variables: ", len(df_drop_forbidden.columns))
df_drop = df_drop_forbidden.drop(*['DepTime','CRSDepTime', 'FlightNum'])
df_drop.printSchema()

#%%
"1- Turn integer into string types to convert them in categorical variables: Year, Month, DayofMonth, DayOfWeek"
"The numerical values that are going to be converted into categorical are the ones in change_type and Year column will be if "
"there is more than 1 distinct value for this column, otherwise, having just 1 value, it will be removed as it does not provide information for the model."
change_type = ["Month", "DayofMonth", "DayOfWeek"]
if df_drop.select('Year').distinct().count() == 1:
    df_drop = df_drop.drop(*['Year'])
    print("In this case there is just one Year value, so this column has been removed from the data frame.")
else:
    change_type.append('Year')
    print("There are more than one value for the Year column, so it will be turned into categorical")
#ampliar
print("The numerical variables turning into categorical are:")
df_cat=df_drop
for col in range(0, len(df_cat.columns)):
    name_col=df_cat.columns[col]
    if name_col in change_type:
        print(name_col)
        df_cat = df_cat.withColumn(name_col, df_cat[name_col].cast(StringType()))
        
print("Therefore, the type of each variable now is:")
df_cat.printSchema()       
#%%
"2- Replace NA por null for Spark to recognize them"
df_filtered=df_cat.filter(df_cat.ArrDelay != "NA")
df_filtered.show()
print("Replacing missing values for null...")
df_rep = df_filtered.replace(("NA"), None)
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
    Counts number of nulls in each column
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
int_type=["ArrDelay", "DepDelay", "Distance"]
from pyspark.sql.types import IntegerType

for i in int_type:
    df_dropped_missing = df_dropped_missing.withColumn(i, df_dropped_missing[i].cast(IntegerType()))
df_dropped_missing.printSchema()
"6- Fill null values in the variables that have not been removed"
catCols = [x for (x, dataType) in df_dropped_missing.dtypes if dataType =="string"]
numCols = [ x for (x, dataType) in df_dropped_missing.dtypes if dataType !="string"]
print("After removing the columns that contain more than 70% of missing values:")
print("Categorical variables: ", catCols)
print("Numerical variables: ", numCols)
print("")
print("These columns are going to be filled as follows:")
print("- Categorical values: with 0.")
print("- Numerical values: with the mean of the column.")
print("")
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
"CORRELATION"
intCols = [x for (x, dataType) in clean_df.dtypes if dataType =="int"]
for i in intCols:
    print(i)
    print(clean_df.stat.corr(i,"ArrDelay"))
    
#%%

#%%
"8- One-hot encoding for categorical variables"
print("Transforming categorical variables using one-hot encoding...")
catCols = [x for (x, dataType) in clean_df.dtypes if dataType =="string"]
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

notStringCols = [x for (x, dataType) in df_encoded.dtypes if ((dataType !="string") and (dataType !="double"))]
df_encoded_clean = df_encoded.select([col for col in notStringCols])
print("Let's see how the data set looks like after one-hot encoding:")
df_encoded_clean.printSchema()

#%%
"9- Splitting the data set in training and test subsets"
train, test = df_encoded_clean.randomSplit([0.7, 0.3], seed=7)
print("Splitting the data set in training and test subsets with 70% and 30% respectively...")
print("Train set length: ", train.count())
print("Test set length: ", test.count())

label_train = train.ArrDelay
train=train.drop(*["ArrDelay"])

label_test = test.ArrDelay
test=test.drop(*["ArrDelay"])
#%%
#integer_cols = [x for (x, dataType) in train.dtypes if (dataType =="int")]
#%%
"10- Scaling numerical variables"

integer_cols = [x for (x, dataType) in train.dtypes if (dataType =="int")]

unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
train_scaled=train
test_scaled=test
print("Scaling values in these variables...")
for i in integer_cols:
    print(i)
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = MinMaxScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    
    train_scaled = pipeline.fit(train_scaled).transform(train_scaled).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i,i+"_Vect")                                                                
    test_scaled = pipeline.fit(test_scaled).transform(test_scaled).withColumn(i+"_Scaled", unlist(i+"_Scaled")).drop(i,i+"_Vect")


   


#%%
"MACHINE LEARNING MODEL"
"MACHINE LEARNING VALIDATION"
