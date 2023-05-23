#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # **Part I: Research Question**
# 
# ## Research Question
# 
# My dataset for this predictive modeling exercise includes data on an internet service provider’s current and former subscribers, with an emphasis on customer churn (whether customers are maintaining or discontinuing their subscription to the ISP’s service).  Data analysis performed on the dataset will be aimed with this research question in mind: is there a relationship between customer lifestyle, or “social” factors, and customer churn?  Lifestyle and social factors might include variables such as age, income, and marital status, among others.

# ---
# 
# ## Objectives and Goals
# 
# Conclusions gleaned from analysis of this data can benefit stakeholders by revealing information on which customer populations may be more likely to “churn”, or to terminate their service contract with the ISP.  Such information may be used to fuel targeted advertising campaigns, special promotional offers, and other strategies related to customer retention.

# ---
# 
# # **Part II: Method Justification**
# 
# ## Assumptions of a multiple regression model
# 
# The assumptions of a multiple regression model are as follows:
# 
# - There exists a linear relationship between each predictor variable and the response variable
# - None of the predictor variables are highly correlated with each other
# - The observations are independent
# - The residuals have constant variance at every point in the linear model
# - The residuals of the model are normally distributed
# 
# For each of these assumptions that is violated, the potential reliability of the multiple regression model decreases.  Adherance to these assumptions can be measured via tests such as Durbin-Watson, Q-Q plot, and VIF (Zach, 2021).

# ---
# 
# ## Tool Selection
# 
# All code execution was carried out via Jupyter Lab, using Python 3.  I used Python as my selected programming language due to prior familiarity and broader applications when considering programming in general.  R is a very strong and robust language tool for data analysis and statistics but finds itself somewhat limited to that niche role (Insights for Professionals, 2019).  I utilized the NumPy, Pandas, and Matplotlib libraries to perform many of my data analysis tasks, as they are among the most popular Python libraries employed for this purpose and see widespread use.  Seaborn is included primarily for its better-looking boxplots, seen later in this document (Parra, 2021).  
# 
# Beyond these libraries, I relied upon the Statsmodels library; in particular its main API, formula API, and the variance_inflation_factor from its outliers_influence module.  Statsmodels is one of several Python libraries that support linear regression.  I am most familiar with it due to the course material's heavy reliance upon it.

# In[3]:


# Imports and housekeeping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set_theme(style="darkgrid")


# ---
# 
# ## Why Multiple Regression?
# 
# Simple linear regression allows us to determine whether a relationship exists between a dependent variable and a single independent variable.  This type of model does have its uses and proper applications, but results in a more simple predictive model without taking into account how other variables may relate to both the independent and dependent variable.  In both the real world and business world it may be rare to encounter data collections with only 2 variables, and relying heavily on simple linear regression models can create a situation where predictions are somewhat unreliable.  Utilizing multiple independent variables in a predictive model can make our predictions stronger and allows higher conviction in the reliance on those models for decision making.
# 

# ---
# 
# # **Part III: Data Preparation**
# 
# ## Data Preparation Goals and Data Manipulations
# 
# I would like my data to include only variables relevant to my research question, and to be clean and free of missing values and duplicate rows.  It will also be important to re-express any categorical variable types with numeric values.  My first steps will be to import the complete data set and execute functions that will give me information on its size, the data types of its variables, and a peek at the data in table form.  I will then narrow the data set to a new dataframe containing only the variables I am concerned with, and then utilizing functions to determine if any null values or duplicate rows exist.

# In[4]:


# Import the main dataset
df = pd.read_csv('churn_clean.csv',dtype={'locationid':np.int64})


# In[5]:


# Display dataset info
df.info()


# In[6]:


# Display dataset top 5 rows
df.head()


# In[7]:


# Trim dataset to variables relevant to research question
columns = ['Area', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'Churn', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']
df_data = pd.DataFrame(df[columns])


# In[8]:


# Check data for null or missing values
df_data.isna().any()


# In[9]:


# Check data for duplicated rows
df_data.duplicated().sum()


# ---
# 
# ## Summary Statistics
# 
# I can use the describe() function to display the summary statistics for the entire dataframe, as well as each variable I'll be evaluating for inclusion in the model.  I have selected the Bandwidth_GB_Year variable as my dependent variable.
# 
# I will also utilize histogram plots to illustrate the distribution of each numeric variable in the dataframe, and countplots for the categorical variables.

# In[10]:


# Display summary statistics for entire dataset - continuous variables
df_data.describe()


# In[11]:


# Display summary statistics for entire dataset - categorical variables
df_data.describe(include = object)


# In[12]:


# Initialize figure size settings
plt.rcParams['figure.figsize'] = [10, 10]


# In[13]:


# Display histogram plots for distribution of continuous variables
df_data.hist()


# In[14]:


# Display histogram plot and summary statistics for Bandwidth_GB_Year
df_data['Bandwidth_GB_Year'].hist(legend = True)
plt.show()
df_data['Bandwidth_GB_Year'].describe()


# In[15]:


# Display histogram plot and summary statistics for Children
df_data['Children'].hist(legend = True)
plt.show()
df_data['Children'].describe()


# In[16]:


# Display histogram plot and summary statistics for Age
df_data['Age'].hist(legend = True)
plt.show()
df_data['Age'].describe()


# In[17]:


# Display histogram plot and summary statistics for Income
df_data['Income'].hist(legend = True)
plt.show()
df_data['Income'].describe()


# In[18]:


# Display histogram plot and summary statistics for Outage_sec_perweek
df_data['Outage_sec_perweek'].hist(legend = True)
plt.show()
df_data['Outage_sec_perweek'].describe()


# In[19]:


# Display histogram plot and summary statistics for Yearly_equip_failure
df_data['Yearly_equip_failure'].hist(legend = True)
plt.show()
df_data['Yearly_equip_failure'].describe()


# In[20]:


# Display histogram plot and summary statistics for Tenure
df_data['Tenure'].hist(legend = True)
plt.show()
df_data['Tenure'].describe()


# In[21]:


# Display histogram plot and summary statistics for MonthlyCharge
df_data['MonthlyCharge'].hist(legend = True)
plt.show()
df_data['MonthlyCharge'].describe()


# In[22]:


# Display countplots for distribution of categorical variables
fig, ax = plt.subplots(figsize = (20,20), ncols = 2, nrows = 2)
sns.countplot(x='Area', data=df_data, ax = ax[0][0])
sns.countplot(x='Marital', data=df_data, ax = ax[0][1])
sns.countplot(x='Gender', data=df_data, ax = ax[1][0])
sns.countplot(x='Churn', data=df_data, ax = ax[1][1])


# In[23]:


# Display countplot and summary statistics for Area
sns.countplot(x='Area', data=df_data)
plt.show()
df_data['Area'].describe()


# In[24]:


# Display countplot and summary statistics for Marital
sns.countplot(x='Marital', data=df_data)
plt.show()
df_data['Marital'].describe()


# In[25]:


# Display countplot and summary statistics for Gender
sns.countplot(x='Gender', data=df_data)
plt.show()
df_data['Gender'].describe()


# In[26]:


# Display countplot and summary statistics for Churn
sns.countplot(x='Churn', data=df_data)
plt.show()
df_data['Churn'].describe()


# ---
# 
# ## Further Preparation Steps
# 
# I will make some adjustments to my data types to make my variables easier to work with.  Conversion of "object" types as "category" in particular will lend itself to more efficient conversion of categorical variables to numeric.

# In[27]:


# Reassign data types
for col in df_data:
    if df_data[col].dtypes == 'object':
        df_data[col] = df_data[col].astype('category')
    if df_data[col].dtypes == 'int64':
        df_data[col] = df_data[col].astype(int)
    if df_data[col].dtypes == 'float64':
        df_data[col] = df_data[col].astype(float)


# In[28]:


# Display dataset info and observe data type changes
df_data.info()


# ---
# 
# Here I will use the cat.codes accessor to perform label encoding on three of my categorical variables.

# In[29]:


# Use cat.codes for label encoding of 4 categorical variables
df_data['Area_cat'] = df_data['Area'].cat.codes
df_data['Marital_cat'] = df_data['Marital'].cat.codes
df_data['Gender_cat'] = df_data['Gender'].cat.codes
df_data['Churn_cat'] = df_data['Churn'].cat.codes


# In[30]:


# Display dataset top 5 rows from label encoded variables
df_data[['Area', 'Marital', 'Gender', 'Churn', 'Area_cat', 'Marital_cat', 'Gender_cat', 'Churn_cat']].head()


# ---
# 
# ## Univariate and Bivariate Visualizations
# 
# Univariate analysis of each variable can be seen above in section 2 of part III, "Data Preparation".  I will make use of Seaborn's regplot() function for bivariate analysis of continuous variables, and Seaborn's boxplot() function for the categorical variables.  Each independent variable is paired against my dependent variable, "Bandwidth_GB_Year".

# In[31]:


# Display regplots for bivariate statistical analysis of continuous variables - dependent variable = Bandwidth_GB_Year
fig, ax = plt.subplots(figsize = (20,20), ncols = 3, nrows = 3)
sns.regplot(x="Children",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[0][0],
            ci=None)
sns.regplot(x="Age",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[0][1],
            ci=None)
sns.regplot(x="Income",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[0][2],
            ci=None)
sns.regplot(x="Outage_sec_perweek",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[1][0],
            ci=None)
sns.regplot(x="Yearly_equip_failure",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[1][1],
            ci=None)
sns.regplot(x="Tenure",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[1][2],
            ci=None)
sns.regplot(x="MonthlyCharge",
            y="Bandwidth_GB_Year",
            data=df_data,
            ax = ax[2][0],
            ci=None)


# In[32]:


# Display boxplots for bivariate analysis of categorical variables - dependent variable = Bandwidth_GB_Year
fig, ax = plt.subplots(figsize = (20, 20), ncols = 2, nrows = 2)
sns.boxplot(x = 'Area_cat', y = 'Bandwidth_GB_Year', data = df_data, ax = ax[0][0])
sns.boxplot(x = 'Marital_cat', y = 'Bandwidth_GB_Year', data = df_data, ax = ax[0][1])
sns.boxplot(x = 'Gender_cat', y = 'Bandwidth_GB_Year', data = df_data, ax = ax[1][0])
sns.boxplot(x = 'Churn_cat', y = 'Bandwidth_GB_Year', data = df_data, ax = ax[1][1])


# ---
# 
# ## Copy of Prepared Data Set
# 
# Below is the code used to export the prepared data set to csv format.

# In[33]:


# Export prepared dataframe to csv
df_data.to_csv(r'C:\Users\wstul\d208\churn_clean_perpared.csv')


# ---
# 
# # **Part IV: Model Comparison and Analysis**
# 
# ## Initial Multiple Regression Model
# 
# Below I will create an initial multiple regression model and display its summary info.

# In[34]:


# Create initial model and display summary
mdl_bandwidth_vs_all = ols("Bandwidth_GB_Year ~ Area_cat + Children + Age + Income + Marital_cat + Gender_cat + Churn_cat + \
                        Outage_sec_perweek + Yearly_equip_failure + MonthlyCharge + Tenure", data=df_data).fit()
print(mdl_bandwidth_vs_all.summary())


# ---
# 
# ## Reducing the Initial Model
# 
# The initial model has a high r-squared score as is, but this is typically the case when a high number of independent variables are included.  I will aim to reduce the model by eliminating variables not suitable for this multiple regression model, using statistical analysis in my selection process.
# 
# To begin I will look at some metrics for the current model.

# In[35]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for initial model
mse_all = mdl_bandwidth_vs_all.mse_resid
print('MSE of original model: ', mse_all)
rse_all = np.sqrt(mse_all)
print('RSE of original model: ', rse_all)
print('Rsquared of original model: ', mdl_bandwidth_vs_all.rsquared)
print('Rsquared Adjusted of original model: ', mdl_bandwidth_vs_all.rsquared_adj)


# ---
# 
# As I proceed through my reduction process, my aim will be to keep r-squared and residual standard error scores close to the initial model's performance.  Higher r-squared scores are considered better, while lower RSE scores are preferable.
# 
# First I will perform a variance inflation factor analysis for all features currently in the model.

# In[36]:


# Perform variance inflation factor analysis for initial feature set
X = df_data[['Area_cat', 'Children', 'Age', 'Income', 'Marital_cat', 'Gender_cat', 'Churn_cat', 'Outage_sec_perweek', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge']]
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# Right away, I can see very high VIF scores for two variables, MonthlyCharge and Outage_sec_perweek.  High VIF values (usually greater than 5) indicate a high degree of multicollinearity with other variables in the model.  This reduces the model accuracy, so I will drop these two variables from the set and repeat my VIF analysis.

# In[37]:


# Drop 2 high VIF variables
X = X.drop(['Outage_sec_perweek', 'MonthlyCharge'], axis = 1)


# In[38]:


# Perform variance inflation factor analysis for trimmed feature set
vif_data = pd.DataFrame()
vif_data['IndVar'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# ---
# 
# Though the Age variable is higher than the other variables after my second VIF analysis, it is below 5 and so merits inclusion as I continue.
# 
# Next I will create a dataframe based on my remaining variables and generate a correlation table to check for other signs of collinearity.

# In[39]:


# Create temporary dataframe of trimmed feature set + dependent variable and display correlation table
df_data_reduced = pd.DataFrame(df_data[['Area_cat', 'Children', 'Age', 'Income', 'Marital_cat', 'Gender_cat', 'Churn_cat', 'Yearly_equip_failure', 'Tenure', 'Bandwidth_GB_Year']])
df_data_reduced.corr()


# ---
# 
# In my correlation table, "Tenure" shares a high collinearity with "Bandwidth_GB_Year", but this is acceptable as "Bandwidth_GB_Year" is our dependent variable.  No other instances of high collinearity appear.
# 
# I will create a reduced model based on my remaining variables to see how our statistics look.

# In[40]:


# Create first reduced model and display summary
mdl_bandwidth_vs_reduced = ols("Bandwidth_GB_Year ~ Area_cat + Children + Age + Income + Marital_cat + Gender_cat + Churn_cat + \
                        Yearly_equip_failure + Tenure", data=df_data).fit()
print(mdl_bandwidth_vs_reduced.summary())


# In[41]:


# Display MSE, RSE for first reduced model
mse_reduced = mdl_bandwidth_vs_reduced.mse_resid
print('MSE of reduced model: ', mse_reduced)
rse_reduced = np.sqrt(mse_reduced)
print('RSE of reduced model: ', rse_reduced)


# ---
# 
# According to this summary, several variables exhibit high p-values, indicating no relationship between that variable and the dependent variable, "Bandwidth_GB_Year".  A value greater than .05 is considered high.  I will remove these variables from the model and once again evaluate the resulting summary and statistics.

# In[42]:


# Create second reduced model and display summary
mdl_bandwidth_vs_features = ols("Bandwidth_GB_Year ~ Children + Age + Marital_cat + Gender_cat + Churn_cat + Tenure", data=df_data).fit()
print(mdl_bandwidth_vs_features.summary())


# In[43]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for second reduced model
mse_features = mdl_bandwidth_vs_features.mse_resid
print('MSE of reduced model: ', mse_features)
rse_features = np.sqrt(mse_features)
print('RSE of reduced model: ', rse_features)
print('Rsquared of reduced model: ', mdl_bandwidth_vs_features.rsquared)
print('Rsquared Adjusted of reduced model: ', mdl_bandwidth_vs_features.rsquared_adj)


# ---
# 
# ## Final Reduced Multiple Regression Model
# 
# At this point, I have eliminated any sources of multicollinearity and collinearity as well as variables exhibiting p-values that exceed .05.  I will finalize the reduced model and check to see how it compares to my initial model which included all variables in the set.

# In[44]:


# Create final reduced model and display summary
mdl_bandwidth_vs_features_final = ols("Bandwidth_GB_Year ~ Children + Age + Marital_cat + Gender_cat + Churn_cat + Tenure", data=df_data).fit()
print(mdl_bandwidth_vs_features_final.summary())


# In[45]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for initial model and final reduced model for comparison
print('MSE of original model: ', mse_all)
print('RSE of original model: ', rse_all)
print('Rsquared of original model: ', mdl_bandwidth_vs_all.rsquared)
print('Rsquared Adjusted of original model: ', mdl_bandwidth_vs_all.rsquared_adj)
mse_final = mdl_bandwidth_vs_features_final.mse_resid
print('MSE of final model: ', mse_final)
rse_final = np.sqrt(mse_final)
print('RSE of final model: ', rse_final)
print('Rsquared of final model: ', mdl_bandwidth_vs_features_final.rsquared)
print('Rsquared Adjusted of final model: ', mdl_bandwidth_vs_features_final.rsquared_adj)


# ---
# 
# ## Data Analysis Process
# 
# During my variable selection process I relied upon trusted methods for identifying variables unsuitable for the model, such as VIF, a correlation table, and p-values.  I measured each model's performance by its r-squared and adjusted r-squared scores, as well as the residual standard error.
# 
# Residual plots for each remaining variable, as well as a Q-Q plot for the entire model are shown below.

# In[46]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Children', fig=fig)


# In[47]:


# Plots for independent variable Tenure
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Tenure', fig=fig)


# In[48]:


# Plots for independent variable Age
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Age', fig=fig)


# In[49]:


# Plots for independent variable Churn_cat
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Churn_cat', fig=fig)


# In[50]:


# Plots for independent variable Marital_cat
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Marital_cat', fig=fig)


# In[51]:


# Plots for independent variable Gender_cat
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_bandwidth_vs_features_final, 'Gender_cat', fig=fig)


# In[52]:


# Q-Q plot for final model
sm.qqplot(mdl_bandwidth_vs_features_final.resid, line='s')


# ---
# 
# # **Part V: Data Summary and Implications**
# 
# ## Summary of Findings
# 
# The regression equation for the final reduced model is as follows:
# 
# **Bandwidth_GB_Year ~ Children + Age + Marital_cat + Gender_cat + Churn_cat + Tenure**
# 
# The coefficients for each variable included:
# 
# **Children       30.5109**
# 
# **Age            -3.3111**
# 
# **Marital_cat    -4.0750**
# 
# **Gender_cat     53.1699**
# 
# **Churn_cat     260.0266**
# 
# **Tenure         84.1205**
# 
# We can use these coefficients to determine the effect each variable will have on the amount of bandwidth use by the customer per year, in GB.  For example, for each additional child in the household, bandwidth will increase by 30.5109 GB.
# 
# The model can provide significant data when evaluating customer retention from a practical perspective, as customers who use the service more (and use more bandwidth) may be more likely to find the service has value and continue using it.  The limitations of using multiple regression models for pratical purposes are always present, however.  Data is based on a sample size, and therefore may not reflect general population as accurately as we would like.  It also assumes correlation is causation, or that when one thing is true it causes another to be true as well.

# ---
# 
# ## Recommended Course of Action
# 
# There are a few key takeaways based on the analysis of this model.  Customers who have been with the service provider for a long use more bandwidth and are likely happy with their service.  They may have even purchased addons such as streaming video or phone.  Based on this, newer customers may be an opportunistic target for special promotions or proactive support initiatives.  At the same time, as a customer ages or has a change in marital status, their usage decreases.  New products or services that cater to those populations may enhance their experience and generate value for them as users of the service.

# ---
# 
# # **Part VI: Demonstration**
# 
# **Panopto Video Recording**
# 
# A link for the Panopto video has been provided separately.  The demonstration includes the following:
# 
# •  Demonstration of the functionality of the code used for the analysis
# 
# •  Identification of the version of the programming environment
# 
# •  Comparison of the two multiple regression models you used in your analysis
# 
# •  Interpretation of the coefficients
# 

# ---
# 
# # **Web Sources**
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
# 
# https://www.geeksforgeeks.org/how-to-create-a-residual-plot-in-python/
# 
# https://www.sfu.ca/~mjbrydon/tutorials/BAinPy/09_regression.html
# 
# https://pbpython.com/categorical-encoding.html
# 
# 

# ---
# 
# # **References**
# 
# 
# Insights for Professionals. (2019, February 26). *5 Niche Programming Languages (And Why They're Underrated).* https://www.insightsforprofessionals.com/it/software/niche-programming-languages
# 
# 
# Parra, H.  (2021, April 20).  *The Data Science Trilogy.*  Towards Data Science.  https://towardsdatascience.com/the-data-science-trilogy-numpy-pandas-and-matplotlib-basics-42192b89e26
# 
# 
# Zach.  (2021, November 16).  *The Five Assumptions of Multiple Linear Regression.*  Statology.  https://www.statology.org/multiple-linear-regression-assumptions/
# 
# 
