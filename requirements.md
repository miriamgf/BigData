# BIG DATA - Practical Application
A model capable of predicting the arrival delay time of a commercial flight, given a set of parameters known at time of take-off.

## CHECKLIST:

- [x] Load the input data, correctly separating all given variables.
_Chosen: 1987. Must work with any subset of the data set._
_Once the data is stored, its location must be provided to the
Spark application as an input parameter. The application should expect to find the data as it is downloaded from the previous
url._

- [ ] Select, process and transform the input variables. Perform variable selection based on some basic analysis and logical criteria.
_Properly handle variable types (numerical, categorical, etc.)_
_Several variables may not contain useful information, are forbidden, contain information difficult to process, provide bettter information when combined with others...
_ArrDelay must not be modified._

- [ ] Basic analysis of each input variable
_Once the pre-processing is done._

- [ ] Machine Learning model - creation
_The students can select any machine learning technique
provided by the Spark API they wish to create this model. This selection must be properly
justified in the report delivered._

- [ ] Machine Learning model - validation and accuracy 
_As in the previous case, the selection of the evaluation technique and accuracy measure must
be sufficiently justified in the report._

IMPORTANT: All data processing, model training and validation must be done with Spark. Use basic MLlib tools for handling the data, training and validating the model.


**GOING FURTHER**
A list of possible things that can be done to improve the quality of the
work and obtain a very good or excellent grade:

- [ ] Smart use of the Spark tools provided to properly read the input file and handle possible
input errors (empty files, missing columns, wrong formats, etc.).
- [ ] Proper exploratory data analysis (possibly including univariate and/or multivariate
analysis) to better understand the input data and provide robust criteria for variable
selection.
- [ ] Smart handling of special format in some input variables, performing relevant processing
and/or transformations.
- [ ] Feature engineering and exploring additional datasets to try to find additional relevant
information.
- [ ] Select more than one valid machine learning algorithm for building the model.
- [ ] Perform model hyper-parameter tuning.
- [ ] Consider more than one possible model performance metric and explain the criteria for
selecting the most appropriate.
- [ ] Use cross-validation techniques to select the best model.
- [ ] Use the full capacities of Spark’s MLlib, including tools for cross-validation,
hyper-parameter tuning, model evaluation, pipelines, etc.
- [ ] Write code that is properly commented, well structured, clean and easy to read.
- [ ] Create an application that can be used with different input files without having to change
a single line of code, both for training and applying the model.
- [ ] Write a report that is both clear and interesting, including insightful analysis and
conclusions.
