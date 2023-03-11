#!/usr/bin/env python
# coding: utf-8

# # WEATHER PREDICTION - USING LINEAR REGRESSION

# In this project, we'll learn how to predict our local weather with machine learning. 
# 
# 1.We'll use python, jupyter notebook, pandas, matplotlib,seaborn,and scikit-learn.
# 
# 2.We'll start by downloading the data, then we'll prepare it for machine learning and try a ridge regression model.

# DATA COLLECTION : downloaded data from https://www.ncdc.noaa.gov/cdo-web/search this website

# ENVIRONMENT : Using Jupyter_notebook
# 
# JupyterLab is the latest web-based interactive development environment for notebooks, code, and data. Its flexible interface allows users to configure and arrange workflows in data science, scientific computing, computational journalism, and machine learning. A modular design invites extensions to expand and enrich functionality.

# A Python library is a collection of related modules. It contains bundles of code that can be used repeatedly in different programs. It makes Python Programming simpler and convenient for the programmer. As we don’t need to write the same code again and again for different programs. Python libraries play a very vital role in fields of Machine Learning, Data Science, Data Visualization, etc.
# 
# IMPORTING LIBRARIES : PANDAS,SEABORN,MATPLOTLIB
# 
# PANDAS : To read CSV file 
# SEABORN & MATPLOTLIB : Used to visualize 

# In[4]:


# Libraries imported 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


#  Loading the CSV into a DataFrame

wth = pd.read_csv("sampledata_weather.csv",index_col="DATE")


# In[6]:


wth


# In[7]:


# loc gets rows (and/or columns) with particular labels. 

wth.loc["2000-01-01":"2003-01-01",:]


# # Preprocessing and visualization

# In[8]:


# identifying the null-(Nan) & NON Nan values from TMAX-column and displaying them in Pie_chart 

explodes = (0,0.3)
plt.pie(wth[wth['NAME']=='OAKLAND INTERNATIONAL AIRPORT, CA US'].TMAX.isna().value_counts(),explode=explodes,startangle=0,colors=['blue','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[9]:


# identifying the null-(Nan) & NON Nan values from TMIN-column and displaying them in Pie_chart 

explodes = (0,0.3)
plt.pie(wth[wth['NAME']=='OAKLAND INTERNATIONAL AIRPORT, CA US'].TMIN.isna().value_counts(),explode=explodes,startangle=0,colors=['red','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[10]:


# identifying the null-(Nan) & NON Nan values from SNOW-column and displaying them in Pie_chart 

explodes = (0,0.3)
plt.pie(wth[wth['NAME']=='OAKLAND INTERNATIONAL AIRPORT, CA US'].SNOW.isna().value_counts(),explode=explodes,startangle=0,colors=['blue','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[11]:


# identifying the null-(Nan) & NON Nan values from TAVG-column and displaying them in Pie_chart 

explodes = (0,0.3)
plt.pie(wth[wth['NAME']=='OAKLAND INTERNATIONAL AIRPORT, CA US'].TAVG.isna().value_counts(),explode=explodes,startangle=0,colors=['Skyblue','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[12]:


# identifying the null-(Nan) & NON Nan values from SNWD-column and displaying them in Pie_chart 

explodes = (0,0.3)
plt.pie(wth[wth['NAME']=='OAKLAND INTERNATIONAL AIRPORT, CA US'].SNWD.isna().value_counts(),explode=explodes,startangle=0,colors=['orange','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[13]:


# Getting percentage of Nan values in each column

wth.apply(pd.isnull).sum()/wth.shape[0]


# According to the Data Documentation we take some core columns and create one new dataframe "core_wth" with specific columns using copy() function.
# changing names of "core_wth" columns intp lower case to access easily.

# In[14]:


core_wth = wth[["NAME","PRCP","SNOW","SNWD","TMAX","TMIN"]].copy()


# In[15]:


core_wth.columns = ["name","precip","snow","snow_depth","temp_max","temp_min"]
core_wth


# In[16]:


# Getting percentage of Nan values in each column

core_wth.apply(pd.isnull).sum()/core_wth.shape[0]


# Getting Value_counts of "snow" column to know how many different values are present.

# In[17]:


core_wth["snow"].value_counts()


# In[18]:


core_wth["snow_depth"].value_counts()


# Deleting "snow" and "snow_depth" columns coz they contain almost Nan values 

# In[19]:


del core_wth["snow"]
del core_wth["snow_depth"]


# In[20]:


# Fetching the null values present in "precip" column using isnull() function 

core_wth[pd.isnull(core_wth["precip"])]


# In[21]:


core_wth["precip"].value_counts()


# In[22]:


# Filling the Nan values with "0" using fillna() function

core_wth["precip"]= core_wth["precip"].fillna(0)


# In[23]:


# Fetching the null values present in "temp_min" column using isnull() function 

core_wth[pd.isnull(core_wth["temp_min"])]


# In[24]:


# Filling the Nan values with "0" using fillna() function

# The ffill() method replaces the NULL values with the value from the previous row 
# (or previous column, if the axis parameter is set to 'columns' ).

core_wth = core_wth.fillna(method="ffill") 
core_wth


# In[25]:


core_wth.apply(pd.isnull).sum()/core_wth.shape[0]


# In[26]:


core_wth.dtypes


# In[27]:


core_wth.index


# In[28]:


#  changing DataType of INDEX from dtype='object' to dtype='datetime64[ns] using pd.to_datetime() function

core_wth.index = pd.to_datetime(core_wth.index)


# In[29]:


core_wth.index


# In[30]:


core_wth.apply(lambda x: (x==9999).sum()) # by lambda function we are chevking that how many columns contain 9999


# In[31]:


core_wth


# # visualization - Preparing for ML

# In[32]:


core_wth[["temp_max","precip"]].hist()


# ANALYSIZING AND VISUALISING THE DATA ACCORDINGLY 

# In[33]:


core_wth[["temp_max","temp_min"]].plot()


# In[34]:


core_wth.index.year.value_counts().sort_index()


# In[35]:


core_wth["precip"].plot()


# In[36]:


core_wth.groupby(core_wth.index.year).sum()["precip"].plot()


# Creating new column "target" by shifting the values of temp_max column using shift(-1).

# In[37]:


core_wth["target"] = core_wth.shift(-1)["temp_max"] 


# In[38]:


core_wth


# In[39]:


core_wth = core_wth.iloc[:-1,:].copy() # loc -fetching strings # iloc - index location
core_wth


# # Plotting barplots and scatter plots 

#  Creating bar plots and scatter plots for the following,all the plots appear in the same plane with
#  x-labels and y-labels evenly spaced.
#  Creating plots that shows the visual representation of "precip", "temp_max", "temp_min",
# "target" with respect to the "name" accordingly.

# In[118]:


fig1 = plt.subplots(2,2, figsize = (20,10))
plt.subplot(2,2,1)
sns.barplot(x = core_wth["temp_max"], y = core_wth["precip"], data = core_wth,hue = core_wth["name"])
plt.title("temp_max vs precip")
plt.xlabel("temp_max")
plt.ylabel("precip") 

plt.subplot(2,2,2)
sns.barplot(x = core_wth["temp_min"], y = core_wth["precip"], data = core_wth,hue = core_wth["name"])
plt.title("temp_min vs precip")
plt.xlabel("temp_min")
plt.ylabel("precip") 

plt.subplot(2,2,3)
sns.scatterplot(x = core_wth["temp_min"], y = core_wth["temp_max"], data = core_wth,hue = core_wth["name"])
plt.title("temp_min vs temp_max")
plt.xlabel("temp_min")
plt.ylabel("temp_max") 


plt.subplot(2,2,4)
sns.scatterplot(x = core_wth["target"], y = core_wth["temp_max"], data = core_wth,hue = core_wth["name"])
plt.title("target vs temp_max")
plt.xlabel("target")
plt.ylabel("temp_max")


# In[119]:


plt.figure(figsize=(30,5))
sns.catplot(x='precip',y ='target',data=core_wth,palette = "magma")
plt.show()


# In[40]:


# Creating line plots for the columns "target","DATE","temp_max" with respect 
# to the "temp_max" and "name" respectively

plt.figure(figsize=(18,8))
sns.set_theme()
sns.lineplot(x = 'target',y='DATE',data=core_wth,palette = "YlOrBr",hue="name" )
plt.xlabel("temp_max",fontweight='bold',size=13,c="red")
plt.ylabel("target",fontweight='bold',size=13)
plt.show()


# In[42]:


# Creating scatter plots for the columns "target","DATE","temp_min" with respect 
# to the "temp_min" and "name" respectively

plt.figure(figsize=(18,8))
sns.set_theme()
sns.scatterplot(x = 'target',y='DATE',data=core_wth,palette = "YlOrBr",color = 'orange',hue = "name" )
plt.xlabel("temp_min",fontweight='bold',size=13,c="red")
plt.ylabel("target",fontweight='bold',size=13)
plt.show()


# In[ ]:


# Creating scatter plots for the columns "target","DATE","precip" with respect 
# to the "precip" and "name" respectively

plt.figure(figsize=(18,8))
sns.set_theme()
sns.lineplot(x = 'target',y='DATE',data=core_wth,palette = "YlOrBr", color='red',hue = "name")
plt.xlabel("precip",fontweight='bold',size=13,c="red")
plt.ylabel("target",fontweight='bold',size=13)
plt.show()


# # Training the model using "RIDGE" function from sklearn.linear_model to reduce the error 

# In[45]:


# Ridge - technique used for reducing the error

from sklearn.linear_model import Ridge  
reg = Ridge(alpha=.1)


# After importing the "Ridge linear_model" from sklearn 
# 1. Creating predictors list.
# 2. Training the machine - accordingly. 
# 3. Testing based on the training data.
# 4. Fitting the regression model to tarin 
# 5. Creating predictions - by fiiting the "reg.predict" to test the predictors.
# 6. Importing mean_absolute_error from sklearn.metrics.
# 7. Create dataframe "combined" by concatenating "test["target"],pd.Series(predictions,index=test.index)".
# 8. Replacing column names of "combined" dataframe as  ["actual","predictions"].
# 9. Plotting and visualizing the "combined" dataframe.
# 10.Pre-processing done - ready to "CREATE MODEL" by defining function and using all code-statements of pre-processing into it.

# In[46]:


predictors = ["precip","temp_max","temp_min"]


# In[47]:


train = core_wth.loc[:"2011-12-31"]


# In[48]:


test = core_wth.loc["2012-01-01":]


# In[49]:


reg.fit(train[predictors],train["target"]) # predictors - x, target -y


# In[50]:


predictions = reg.predict(test[predictors])  # x_test


# In[51]:


# Importing mean_absolute_error from sklearn.metrics - form of an error - gives avg error value.

from sklearn.metrics import mean_absolute_error 


# In[52]:


mean_absolute_error(test["target"],predictions) # y_test  


# In[53]:


combined = pd.concat([test["target"],pd.Series(predictions,index=test.index)],axis=1)


# In[54]:


combined


# In[55]:


combined.columns = ["actual","predictions"]


# In[56]:


combined


# In[57]:


combined.plot()


# In[58]:


reg.coef_ # gives form of an error 


# # Creating  PREDICTION-MODEL using Python Function:
# 
# -->Functions are integral parts of every programming language because they help make your code more modular and reusable.
# 
# -->In Python, you define a function with the def keyword, then write the function identifier (name) followed by parentheses and a colon.
# 
# -->The next thing you have to do is make sure you indent with a tab or 4 spaces, and then specify what you want the function to do for you.

# 1. Creating function named "create_predictions"
# 2. Specifying all pre-processed data in to the function 
# 3. Returning "error" and "combined" dataframe

# In[68]:


def create_predictions(predictors,core_wth,reg):
    train = core_wth.loc[:"2011-12-31"]
    test = core_wth.loc["2012-01-01":]
    reg.fit(train[predictors],train["target"])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"],predictions)
    combined = pd.concat([test["target"],pd.Series(predictions,index=test.index)],axis=1)
    combined.columns = ["actual","predictions"]
    return error,combined 


# In[69]:


# ADDING MONTHLY_AVG column to "core_wth" dataframe


# --> Checking the function by "Creating "monthly_avg" column in "core_wth" dataframe" by using "temp_max".
# 
# --> Adding "monthly_avg" to predictors list.
# 
# --> passing the function - which returns error and combined
# "error,combined = create_predictions(predictors,core_wth,reg)".
# 
# --> visualizing "combined" dataframe.

#  

# In[70]:


core_wth["monthly_avg"] = core_wth["temp_max"].groupby(core_wth.index.month).apply(lambda x: x.expanding(1).mean())


# In[71]:


core_wth


# In[72]:


predictors = ["precip","temp_max","temp_min","monthly_avg"]


# In[73]:


error,combined = create_predictions(predictors,core_wth,reg)


# In[74]:


error


# In[75]:


combined


# In[76]:


combined.plot()


# #  RUNNING MODEL DIAGNOSIS 

# In[77]:


reg.coef_


# In[78]:


core_wth.corr()["target"]


# --> Creating & Adding "difference" column to the "combined" dataframe
# 
# --> Difference = (combined["actual"]- combined["predictions"]).abs()

# In[79]:


combined["diff"] = (combined["actual"]- combined["predictions"]).abs()


# In[80]:


#  Sorting the values of "combined" based on "difference column" when "ascending" is True 

combined.sort_values("diff",ascending = True).head()


# --> visualizing the "combined" df 
# 
# --> It will display "actual","predictions","diff"

# In[81]:


combined.plot()


# # Visualizing combined as Stripplot using "Seaborn"

# In[82]:


sns.color_palette()
sns.stripplot(x = combined.actual,y = combined.predictions)


# # CONCLUSION 

# Weather forecasts are increasingly accurate and useful, and their benefits extend widely to predict the weather. 
# 
# While much has been accomplished in improving weather forecasts using "Ridge_Model using linear regression by function("creating_preductions")".
# 
# 1. We created predictors from the base "Csv_file" using "Pandas".
# 
# 2. We trained the model based on some sort of data using "indexing",
# i.e; we trained the model by giving data starting from "2000-01-01" to "2011-12-31".
# 
# 3. Based on the training we will test the data and make predictions from "2012-01-01" to "2023-02-27" .
# 
# 4. We make predictions by using "Linear Regression - Ridge Model" .
# 
# 5. We visualized and created some plots,charts and lines to understand data clearly when required with the help of libraries "matplotlib & seaborn"
# 
# 5. We got errors between "3.4 to 3.2" by which we can conclude that it is a preferable model to make predictions.There remains much room for improvement. 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




