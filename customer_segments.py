
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:

# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [23, 4, 9]


# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

import seaborn as sns
sns.heatmap((samples-data.mean())/data.std(ddof=0), annot=True, cbar=False, square=True)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# *What kind of establishment (customer) could each of the three samples you've chosen represent?*  
# **Hint:** Examples of establishments include places like markets, cafes, and retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant.

# **Answer:**
# - Analyzing the three samples:
#     - Sample 0:
#         - Fresh: **26373** (> 12000.297727 mean value)
#         - Milk: **36423** (> 5796.265909 mean value)
#         - Grocery **22019** (> 7951.277273 mean value)
#         - Frozen: **5154** (> 3071.931818 mean value)
#         - Detegents_Paper: **4337** (> 2881.493182 mean value)
#         - Delicatessen: **16523** (> 1524.870455 mean value)
#         - *Total Spending*: 110829
#     - Sample 1:
#         - Fresh: **22615** (> 12000.297727 mean value)
#         - Milk: 5410 (< 5796.265909 mean value)
#         - Grocery 7198 (< 7951.277273 mean value)
#         - Frozen: **3915** (> 3071.931818 mean value)
#         - Detegents_Paper: 1777 (< 2881.493182 mean value)
#         - Delicatessen: **5185** (> 1524.870455 mean value)
#         - *Total Spending*: 46100
#     - Sample 2:
#         - Fresh: 6006 (< 12000.297727 mean value)
#         - Milk: **11093** (> 5796.265909 mean value)
#         - Grocery **18881** (> 7951.277273 mean value)
#         - Frozen: 1159 (< 3071.931818 mean value)
#         - Detegents_Paper: **7425** (< 2881.493182 mean value)
#         - Delicatessen: **2098** (< 1524.870455 mean value)
#         - *Total Spending*: 46662
# 
# - The three customers vary significantly from one another. Possible establishments:
#     - Sample 0: retailer (super-market)
#     - sample 1: cafe-restaurant 
#     - sample 2: mini-market

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[6]:

# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data
new_data = new_data.drop(['Grocery'], axis = 1)
target = data.Grocery

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, target, 
                                                    test_size=0.25, random_state=7)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 42)
regressor.fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print score


# ### Question 2
# *Which feature did you attempt to predict? What was the reported prediction score? Is this feature is necessary for identifying customers' spending habits?*  
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data.

# **Answer:**
# - I attempted to predict all the features to see which are more critical. Here are the reported prediction scores for every feature:
#     - r2 score: -0.679625 --> **Fresh** 
#     - r2 score: -2.130021 --> **Milk**
#     - r2 score: +0.529609 --> Grocery
#     - r2 score: -1.145774 --> **Frozen**
#     - r2 score: +0.745188 --> Detergents_Paper
#     - r2 score: -1.068265 --> **Delicatessen**
# 
# - Commenting on the results:
#     - The negative r2 score implies the model fails to fit the data. So the features 'Fresh', 'Milk', 'Frozen' and 'Delicatessen' are very importand, because they contain critical information. These features are **necessary** for identifying customers' spending habits.
#     - On the other hand the model which tries to predict the 'Detergents_Paper' feature is going very good (r2 score = 0.745188). Also, the model which tries to predict the 'Grocery' feature is going good (r2 score = 0.529609). These features **are not so necessary** for identifying customers' spending habits.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[7]:

# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[9]:

import matplotlib.pyplot as plt

corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True, annot=True,
                     cmap='RdBu', fmt='+.3f')
    plt.xticks(rotation=45, ha='center')


# ### Question 3
# *Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?*  
# **Hint:** Is the data normally distributed? Where do most of the data points lie? 

# **Answer:**
# - The more pairs of features doesn't exhibit some degree of correlation. That's why the 4 models of total 6 fail to fit the data. I see that there is some correlation in some pairs of features 'Grocery' and 'Detergents_Paper', as i expected.
# - More specific i see some correlations in the following pairs of features:
#     - Grocery - Milk
#     - Grocery - Detergents_Paper
#     - Detergents_Paper - Milk

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[12]:

# TODO: Scale the data using the natural logarithm
log_data = data.apply(lambda x: np.log(x))

# TODO: Scale the sample data using the natural logarithm
log_samples = samples.apply(lambda x: np.log(x))

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[13]:

# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[18]:

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    IQR = Q3 - Q1
    step = 1.5*IQR
    
    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
                          
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [65, 66, 75, 128, 154]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# ### Question 4
# *Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 

# **Answer:**
# - The data which are considered outliers for more than one feature are the following (with bold are the values of the corresponding features):
#     - 65:
#         - **4.442651** (Fresh)
#         - 9.950323 (Milk)
#         - 10.732651 (Grocery)
#         - **3.583519** (Frozen)
#         - 10.095388 (Detergents_Paper)
#         - 7.260523 (Delicatessen)
#     - 66:
#         - **2.197225** (Fresh)
#         - 7.335634 (Milk)
#         - 8.911530 (Grocery)
#         - 5.164786 (Frozen)
#         - 8.151333 (Detergents_Paper)
#         - **3.295837** (Delicatessen)
#     - 75:
#         - 9.923192 (Fresh)
#         - 7.036148 (Milk)
#         - **1.098612** (Grocery)
#         - 8.390949 (Frozen)
#         - **1.098612** (Detergents_Paper)
#         - 6.882437 (Delicatessen)
#     - 128:
#         - **4.941642** (Fresh)
#         - 9.087834 (Milk)
#         - 8.248791 (Grocery)
#         - 4.955827 (Frozen)
#         - 6.967909 (Detergents_Paper)
#         - **1.098612** (Delicatessen)
#     - 154:
#         - 6.432940 (Fresh)
#         - **4.007333** (Milk)
#         - **4.919981** (Grocery)
#         - 4.317488 (Frozen)
#         - 1.945910 (Detergents_Paper)
#         - **2.079442** (Delicatessen)
#         
# - The presence of outliers can often skew results which take into consideration these data points. On the other hand the model maybe will not be affected from the outliers. In this particular problem we want to find customer segments, So I will be careful and i will remove all the above outliers.
# 
# Helpful links:
# - http://www.theanalysisfactor.com/outliers-to-drop-or-not-to-drop/
# - http://graphpad.com/guides/prism/6/statistics/index.htm?stat_checklist_identifying_outliers.htm

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[21]:

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA(n_components=6).fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# In[22]:

## Extra review code

# create an x-axis variable for each pca component
x = np.arange(1,7)

# plot the cumulative variance
plt.plot(x, np.cumsum(pca.explained_variance_ratio_), '-o', color='black')

# plot the components' variance
plt.bar(x, pca.explained_variance_ratio_, align='center', alpha=0.5)

# plot styling
plt.ylim(0, 1.05)
plt.annotate('Cumulative explained variance',
             xy=(3.7, .88), arrowprops=dict(arrowstyle='->'), xytext=(4.5, .6))
for i,j in zip(x, np.cumsum(pca.explained_variance_ratio_)):
    plt.annotate(str(j.round(4)),xy=(i+.2,j-.02))
plt.xticks(range(1,7))
plt.xlabel('PCA components')
plt.ylabel('Explained Variance')
plt.show()


# ### Question 5
# *How much variance in the data is explained* ***in total*** *by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the indivdual feature weights.

# **Answer:**
# - Explained Variance by principal component:
#     - **0.4430** (1st)
#     - **0.2638** (2nd)
#     - 0.1231 (3rd)
#     - 0.1012 (4th)
#     - 0.0485 (5th)
#     - 0.0204 (6th)
# - Some maths:
#     - Total Explained Variance = 1
#     - Total Explained Variance the first four principal components = 0.9311
#     
# **Note**: The first four components can explain the original data by more than 90% (93.11%).
# 
# - Representations in terms of customer spending:
#     - Dimension 1:
#         - An increase of 'Detergents_Paper', 'Grocery', 'Milk' and 'Delicatessen' corresponds with an increase  of dimension 1.
#         - On the other hand an increase of 'Fresh' and 'Fresh' corresponds with a decrease of dimension 1.
#         - High positive correlation between:
#             - Detergents Paper
#             - Grocery
#             - Milk
#         - Customer spendings:
#             - High value of PC1:
#                 - The customer tends to spend more money in:
#                     - Detergents Paper
#                     - Grocery
#                     - Milk 
#                     - Delicatessen
#             - Low value of PC1:
#                 - The customer tends to spend more money in:
#                     - Frozen
#                     - Fresh
#     - Dimension 2:
#         - An increase of 'Fresh', 'Frozen, 'Delicatessen', 'Milk', 'Grocery' and 'Detergents_Paper' corresponds with an increase  of dimension 2.
#         - High positive correlation between:
#             - Fresh
#             - Frozen
#             - Delicatessen
#         - Customer spendings:
#             - High value of PC1:
#                 - The customer tends to spend more money in:
#                     - Fresh
#                     - Frozen
#                     - Delicatessen
#                     - Milk
#     - Dimension 3:
#         - An increase of 'Delicatessen', 'Frozen and 'Milk' corresponds with an increase  of dimension 3.
#         - On the other hand an increase of 'Fresh',  'Detergents_Paper' and 'Grocery' corresponds with a decrease of dimension 3.
#         - High positive correlation between:
#             - Delicatessen
#             - Frozen
#         - High negative correlation between:
#             - Fresh
#             - Detergents Paper
#         - Customer spendings:
#             - High value of PC1:
#                 - The customer tends to spend more money in:
#                     - Delicatessen
#                     - Frozen
#             - Low value of PC1:
#                 - The customer tends to spend more money in:
#                     - Fresh
#                     - Detergents Paper
#     - Dimension 4:
#         - An increase of 'Frozen', 'Detergents_Paper', 'Grocery' and 'Milk' corresponds with an increase  of dimension 3.
#         - On the other hand an increase of 'Delicatessen' and 'Fresh' corresponds with a decrease of dimension 4.
#         - High positive correlation between:
#             - Frozen
#             - Detergents Paper
#         - High negative correlation berween:
#             - Delicatessen
#             - Fresh
#         - Customer spendings:
#             - High value of PC1:
#                 - The customer tends to spend more money in:
#                     - Frozen
#                     - Detergents Paper
#                     - Grocery
#             - Low value of PC1:
#                 - The customer tends to spend more money in:
#                     - Delicatessen
#                     - Fresh
# 
# 
# - Review comments:
#     - A principal component is an engineered feature made from the original features. The first dimension has 4 features correlated together and to features correlated in the other direction. The sign of the features can not be interpreted. The signs are actually reversible, and if you run it multiple times on your computer you may have noticed this. Have a look at
# http://stats.stackexchange.com/questions/30348/is-it-acceptable-to-reverse-a-sign-of-a-principal-component-scorecorrelated
#     - What we are looking for here is the largest absolute value magnitude features. These are the features that are most heavily represented.
#     - Further reading:
# You can read more about how to interpret the dimensions here:
# https://onlinecourses.science.psu.edu/stat505/node/54
# http://setosa.io/ev/principal-component-analysis/

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[23]:

# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[24]:

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[25]:

# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[26]:

# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# *What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*

# **Answer:**
# - Advantages of K-Means:
#     - Simple and easy to implement
#     - Fast and efficient in terms of computational costs
#     - General-purpose algorithm
# - Advantages of GMM:
#     - It is the fastest algorithm for learning mixture models.
#     - As this algorithm maximizes only the likelihood, it will not bias the means towards zero, or bias the cluster sizes to have specific structures that might or might not apply.
#     - The Mixture of Gaussian model helps us to express this uncertainty.
#     - Source: http://scikit-learn.org/stable/modules/mixture.html#mixture
# 
# Summary: When i see the biplot graph, it not so clear to me which are the clusters. For this reason i will use GMM clustering algorithm. GMM help us to express uncertainty of clustering.  It starts with some prior belief about how certain we are about each point's cluster assignments. As it goes on, it revises those beliefs. But it incorporates the degree of uncertainty we have about our assignment.
# 
# source: https://www.quora.com/What-is-the-difference-between-K-means-and-the-mixture-model-of-Gaussian
# 
# Helpful links:
# - http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/mixture.html
# - http://www.nickgillian.com/wiki/pmwiki.php/GRT/GMMClassifier
# - http://playwidtech.blogspot.hk/2013/02/k-means-clustering-advantages-and.html
# - http://www.improvedoutcomes.com/docs/WebSiteDocs/Clustering/K-Means_Clustering_Overview.htm
# - http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means
# - http://www.r-bloggers.com/k-means-clustering-is-not-a-free-lunch/
# - http://www.r-bloggers.com/pca-and-k-means-clustering-of-delta-aircraft/
# - https://shapeofdata.wordpress.com/2013/07/30/k-means/
# - http://mlg.eng.cam.ac.uk/tutorials/06/cb.pdf
# - https://www.quora.com/What-is-the-difference-between-K-means-and-the-mixture-model-of-Gaussian

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[54]:

# TODO: Apply your clustering algorithm of choice to the reduced data
# from sklearn.cluster import KMeans
from sklearn.mixture import GMM
clusterer = GMM(n_components=2).fit(reduced_data)
# clusterer = KMeans(n_clusters=2).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.means_
# centers = clusterer.cluster_centers_ 

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
from sklearn.metrics import silhouette_score
score = silhouette_score(reduced_data, preds)
print score


# ### Question 7
# *Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 

# **Answer:**
# - 2 clusters: **0.411818864386** (best score)
# - 3 clusters: 0.37513463899
# - 4 clusters: 0.335544911576
# - 5 clusters: 0.295441470747
# - 6 clusters: 0.270498049376
# - 7 clusters: 0.322542962762

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[55]:

# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds, centers, pca_samples)


# In[56]:

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. *What set of establishments could each of the customer segments represent?*  
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`.

# **Answer:**
# - Segment 0:
#     - Fresh: 4316.0 (< 12000.297727 mean value)
#     - Milk: **6347.0** (> 5796.265909 mean value)
#     - Grocery: **9555.0** (> 7951.277273 mean value)
#     - Frozen: 1036.0 (< 3071.931818 mean value)
#     - Detergents_Paper: **3406.0** (> 2881.493182 mean value)
#     - Delicatessen: 945.0 (< 1524.870455 mean value)
#     - *Possible establishment*: market, retailer (especially grocery, milk, detergents paper)
# - Segment 1:
#     - Fresh: 8812.0 (< 12000.297727 mean value)
#     - Milk: 2052.0 (< 5796.265909 mean value)
#     - Grocery: 2689.0 (< 7951.277273 mean value)
#     - Frozen: 2058.0 (< 3071.931818 mean value)
#     - Detergents_Paper: 337.0 (< 2881.493182 mean value)
#     - Delicatessen: 712.0 (< 1524.870455 mean value)
#     - *Possible establishment*: cafe, restaurant (especially fresh)

# ### Question 9
# *For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[57]:

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Answer:**
# - Sample 0: Cluster 0 --> Segment 0 (consistent, general category markets-retailers)
# - Sample 1: Cluster 0 --> Segment 0 (unconsistent, this data point is located near the boundary of separation)
# - Sample 2: Cluster 0 --> Segment 0 (consistent, general category markets-retailers)

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. *How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*  
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:**
# 
# We have 6 different segments after clustering and we want to find the appropriate segments to roll out the new delivery service. In order to do that, we have to use A/B testing with the following way. So for each customer segment (total 6), we can pick a random sample group of data and again randomly split the sample into control and test groups. After this group selection we apply the new delivery service in test group. Finally we use an appropriate metric, customer satisfaction score, to compare the different reaction of control and test group. If the test group is more satisfied than control group we apply the new delivery service, else we don't. After doing this for all segments we will be able to tell the potential impact for all of them. After A/B testing we can procide in further evaluations, such profit margin of the new service.
# 
# source: https://discussions.udacity.com/t/a-b-test-question-what-is-the-point-and-what-are-you-looking-for/173455
# 
# Here are a few links for further reading on A/B testing:
# - https://www.quora.com/When-should-A-B-testing-not-be-trusted-to-make-decisions/answer/Edwin-Chen-1
# - http://multithreaded.stitchfix.com/blog/2015/05/26/significant-sample/
# - http://techblog.netflix.com/2016/04/its-all-about-testing-netflix.html
# - https://vwo.com/ab-testing/
# - http://stats.stackexchange.com/questions/192752/clustering-and-a-b-testing

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# *How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?*  
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:**
# 
# This is a classification problem. We will use the old data as labeled data, using the segments which were created via the clustering. So we will train a classifier and after that we will be able to classify the new 10 data using the estimated product spendings. The target variable of the classifier is one of the two different customer segments. After the end of this procedure we will be able to determine the most appropriate delivery service for the new customer, based on the A/B testing.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[59]:

# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers, pca_samples)


# ### Question 12
# *How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*

# **Answer:**
# 
# In clustering i defined 2 segments. Actually the clustering performed well compered to this underlying distribution oh HoReCa and Retailer customers. In general clusters 0 corresponds to Retailer and the cluster 1 correspond to HoReCa. We see that the clustering make some mistakes in HoReCa data points (classified in Retailer cluster).

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
