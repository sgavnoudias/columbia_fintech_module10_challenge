#!/usr/bin/env python
# coding: utf-8

# # Module 10 Application
# 
# ## Challenge: Crypto Clustering
# 
# In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.
# 
# The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.
# 
# The steps for this challenge are broken out into the following sections:
# 
# * Import the Data (provided in the starter code)
# * Prepare the Data (provided in the starter code)
# * Find the Best Value for `k` Using the Original Data
# * Cluster Cryptocurrencies with K-means Using the Original Data
# * Optimize Clusters with Principal Component Analysis
# * Find the Best Value for `k` Using the PCA Data
# * Cluster the Cryptocurrencies with K-means Using the PCA Data
# * Visualize and Compare the Results

# ### Import the Data
# 
# This section imports the data into a new DataFrame. It follows these steps:
# 
# 1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.
# 
# 2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.
# 
# 
# > **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 

# In[ ]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load the data into a Pandas DataFrame
market_data_df = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
market_data_df.head(10)


# In[ ]:


# Generate summary statistics
market_data_df.describe()


# In[ ]:


# Plot your data to see what's in your DataFrame
market_data_df.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data
# 
# This section prepares the data before running the K-Means algorithm. It follows these steps:
# 
# 1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.
# 
# 2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.
# 

# In[ ]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled_arr = StandardScaler().fit_transform(market_data_df)


# In[ ]:


# Check the type of data structure returned by StandardScalar
display(type(market_data_scaled_arr))


# In[ ]:


# Create a DataFrame with the scaled data
market_data_scaled_df = pd.DataFrame(
    market_data_scaled_arr,
    columns=market_data_df.columns
)


# In[ ]:


# Checkpoint: Interim check of the dataframe and resultant statistics after standard scaling algorithm (i.e. mean ~ 0, std ~ `)
display(market_data_scaled_df.head())
market_data_scaled_df.describe()


# In[ ]:


# Copy the crypto names from the original data
market_data_scaled_df["coin_id"] = market_data_df.index


# In[ ]:


# Checkpoint: Interim check of the dataframe (check coin_id added as a column)
display(market_data_scaled_df.head())


# In[ ]:


# Set the coinid column as index
market_data_scaled_df = market_data_scaled_df.set_index("coin_id")

# Display sample data
market_data_scaled_df.head()


# ---

# ### Find the Best Value for k Using the Original Data
# 
# In this section, you will use the elbow method to find the best value for `k`.
# 
# 1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following question: What is the best value for `k`?

# In[ ]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))   # Range: [1, 11)
display(k)  # checkpoint


# In[ ]:


# Create an empy list to store the inertia values
inertia = []
display(inertia)  # checkpoint


# In[ ]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    kmeans_model = KMeans(n_clusters=i, random_state=0)
    kmeans_model.fit(market_data_scaled_df)
    inertia.append(kmeans_model.inertia_)


# In[ ]:


# Checkpoint, confirm the type of the inertia field and display the contents
display(type(inertia))
display(inertia)


# In[ ]:


# Create a dictionary with the data to plot the Elbow curve
elbow_data_dict = {
    "k": k,
    "inertia": inertia
}


# In[ ]:


display(elbow_data_dict)  # Checkpoint


# In[ ]:


# Create a DataFrame with the data to plot the Elbow curve
elbow_data_df = pd.DataFrame(elbow_data_dict) 


# In[ ]:


display(elbow_data_df)  # Checkpoint


# In[ ]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_data_df.hvplot.line(
    x="k", 
    y="inertia", 
    title="Cryto Market Data Elbow Curve", 
    xticks=k)


# #### Answer the following question: What is the best value for k?
# **Question:** What is the best value for `k`?
# 
# **Answer:** From the elbow curve, it appears that the optimal value for k, the nubmer of clusters, is 4 (best elbow point in the curve).  Higher values of k do not give you any significant reduction in the intertia (variance) 

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data
# 
# In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the original data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.
# 
# 4. Create a copy of the original data and add a new column with the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[ ]:


# Initialize the K-Means model using the best value for k
kmeans_model = KMeans(n_clusters=4)


# In[ ]:


display(kmeans_model)  # Checkpoint


# In[ ]:


# Fit the K-Means model using the scaled data
kmeans_model.fit(market_data_scaled_df)


# In[ ]:


# Predict the clusters to group the cryptocurrencies using the scaled data
market_data_scaled_cluster_segments_arr = kmeans_model.predict(market_data_scaled_df)

# View the resulting array of cluster values.
print(market_data_scaled_cluster_segments_arr)


# In[ ]:


# Create a copy of the DataFrame
market_data_scaled_predict_df = market_data_scaled_df.copy()


# In[ ]:


display(market_data_scaled_predict_df.head())  # Checkpoint


# In[ ]:


# Add a new column to the DataFrame with the predicted clusters
market_data_scaled_predict_df["Cluster Segment"] = market_data_scaled_cluster_segments_arr

# Display sample data
display(market_data_scaled_predict_df.head()) 


# In[ ]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
market_data_scaled_predict_df.hvplot.scatter(
    width = 1000,
    height = 500,
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d", 
    by="Cluster Segment",
    hover_cols = ["coin_id"],
    title = "Scatter Plot of Cryto Market Data Cluster Segments - k=4")


# ---

# ### Optimize Clusters with Principal Component Analysis
# 
# In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.
# 
# 1. Create a PCA model instance and set `n_components=3`.
# 
# 2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 
# 
# 3. Retrieve the explained variance to determine how much information can be attributed to each principal component.
# 
# 4. Answer the following question: What is the total explained variance of the three principal components?
# 
# 5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.

# In[ ]:


# Create a PCA model instance and set `n_components=3`.
pca_model = PCA(n_components=3)


# In[ ]:


display(pca_model)  # Checkpoint


# In[ ]:


# Check DataFrame data types (confirm all data are numeric)
market_data_scaled_df.dtypes


# In[ ]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
market_data_scaled_pca_arr = pca_model.fit_transform(market_data_scaled_df)

# View the first five rows of the DataFrame. 
market_data_scaled_pca_arr[:5]


# In[ ]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
display(pca_model.explained_variance_ratio_)                  


# #### Answer the following question: What is the total explained variance of the three principal components?
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** (See print result in next cell)

# In[ ]:


# Answer
print(f"The 1st principal component contains ~{(pca_model.explained_variance_ratio_[0]*100):0.2f}% of the variance; "
      f"the 2nd princiapl component contains ~{(pca_model.explained_variance_ratio_[1]*100):0.2f}% of the variance; "
      f"the 3rd principal component contains ~{(pca_model.explained_variance_ratio_[2]*100):0.2f}% of the variance;")
print(f"All 3 components together contain ~{((pca_model.explained_variance_ratio_[0]+pca_model.explained_variance_ratio_[1]+pca_model.explained_variance_ratio_[2])*100):0.2f}% of the original information.")


# In[ ]:


# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
market_data_scaled_pca_df = pd.DataFrame(
    market_data_scaled_pca_arr,
    columns=["PC1", "PC2", "PC3"])


# In[ ]:


display(market_data_scaled_pca_df.head())  # Checkpoint


# In[ ]:


# Copy the crypto names from the original data
market_data_scaled_pca_df["coin_id"] = market_data_scaled_df.index


# In[ ]:


display(market_data_scaled_pca_df.head())  # Checkpoint


# In[ ]:


# Set the coinid column as index
market_data_scaled_pca_df = market_data_scaled_pca_df.set_index("coin_id")


# In[ ]:


# Display sample data
display(market_data_scaled_pca_df.head())


# ---

# ### Find the Best Value for k Using the PCA Data
# 
# In this section, you will use the elbow method to find the best value for `k` using the PCA data.
# 
# 1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 
# 
# 2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.
# 
# 3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?

# In[ ]:


# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))   # Range: [1, 11)


# In[ ]:


# Create an empy list to store the inertia values
inertia = []


# In[ ]:


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    kmeans_model = KMeans(n_clusters=i, random_state=0)
    kmeans_model.fit(market_data_scaled_pca_df)
    inertia.append(kmeans_model.inertia_)


# In[ ]:


# Checkpoint, confirm the type of the inertia field and display the contents
display(type(inertia))
display(inertia)


# In[ ]:


# Create a dictionary with the data to plot the Elbow curve
# Create a dictionary with the data to plot the Elbow curve
elbow_pca_dict = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
elbow_pca_df = pd.DataFrame(elbow_pca_dict)


# In[ ]:


display(elbow_pca_df)  # Checkpoint


# In[ ]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_pca_df.hvplot.line(
    x="k", 
    y="inertia", 
    title="Cryto Market Data PCA Elbow Curve", 
    xticks=k)


# #### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** From the elbow curve, it appears that the optimal value for k, the nubmer of clusters, is 4 (best elbow point in the curve).  Higher values of k do not give you any significant reduction in the intertia (variance) 
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** No, the PCA best k value of 4 does not differ from above determined original data best k value (which is expected since the PCA approached close to 90% of the original information, so therefore, # of clusters to best fit the data is expected to be the same)

# ---

# ### Cluster Cryptocurrencies with K-means Using the PCA Data
# 
# In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.
# 
# 1. Initialize the K-Means model with four clusters using the best value for `k`. 
# 
# 2. Fit the K-Means model using the PCA data.
# 
# 3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.
# 
# 4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.
# 
# 5. Create a scatter plot using hvPlot by setting `x="PC1"` and `y="PC2"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.

# In[ ]:


# Initialize the K-Means model using the best value for k
kmeans_model = KMeans(n_clusters=4)


# In[ ]:


# Fit the K-Means model using the PCA data
kmeans_model.fit(market_data_scaled_pca_df)


# In[ ]:


# Predict the clusters to group the cryptocurrencies using the PCA data
market_data_scaled_pca_cluster_segments_arr = kmeans_model.predict(market_data_scaled_pca_df)

# View the resulting array of cluster values.
print(market_data_scaled_pca_cluster_segments_arr)


# In[ ]:


# Create a copy of the DataFrame with the PCA data
market_data_scaled_pca_predict_df = market_data_scaled_pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
market_data_scaled_pca_predict_df["Cluster Segment"] = market_data_scaled_pca_cluster_segments_arr

# Display sample data
display(market_data_scaled_pca_predict_df.head())  # Checkpoint


# In[ ]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
market_data_scaled_pca_predict_df.hvplot.scatter(
    width = 1000,
    height = 500,
    x="PC1", 
    y="PC2", 
    by="Cluster Segment",
    hover_cols = ["coin_id"],
    title = "Scatter Plot of Cryto Market Data PCA Cluster Segments - k=4")


# ---

# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.
# 
# 1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.
# 
# 2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.
# 
# 3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
# > **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).

# In[ ]:


# Composite plot to contrast the Elbow curves

# First, assign the original data elbow curve data frame to a plot variable
elbow_data_df_plot = elbow_data_df.hvplot.line(
    x="k", 
    y="inertia", 
    color="blue", 
    title="Cryto Market Data Overlay Elbow Curve", 
    xticks=k)

# Second, assign the cluster data elbow curve data frame to a plot variable
elbow_pca_df_plot = elbow_pca_df.hvplot.line(
    x="k", 
    y="inertia", 
    color="red", 
    title="Cryto Market PCA Overlay Elbow Curve", 
    xticks=k)

# Plot using the (+) to plot a composite view/plot
elbow_data_df_plot + elbow_pca_df_plot


# In[ ]:


# Compoosite plot to contrast the clusters

# First, assign the original data cluster data frame to a plot variable
market_data_scaled_predict_df_plot = market_data_scaled_predict_df.hvplot.scatter(
    width = 800,
    height = 500,
    x="price_change_percentage_24h", 
    y="price_change_percentage_7d", 
    by="Cluster Segment",
    hover_cols = ["coin_id"],
    title = "Scatter Plot of Cryto Market Data Cluster Segments - k=4")

# Second, assign the pca cluster data frame to a plot variable
market_data_scaled_pca_predict_df_plot = market_data_scaled_pca_predict_df.hvplot.scatter(
    width = 800,
    height = 500,
    x="PC1", 
    y="PC2", 
    by="Cluster Segment",
    hover_cols = ["coin_id"],
    title = "Scatter Plot of Cryto Market Data PCA Cluster Segments - k=4")

# Plot using the (+) to plot a composite view/plot
market_data_scaled_predict_df_plot + market_data_scaled_pca_predict_df_plot


# #### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:**= Using the PCA method to cluster the data did a fairly nice job of "clustering" same data together.  Although visually, it looks very different - but this is expected since the resultant PCA data is "transformed" from the original data set, the combined data groups in the cluster are similar.  You can notice that in both methods, the majority of the points are clustered into 2 groups (cluster 0 and 3 in the original data, and cluster 0 and 2 in the pca cluster data).  To prove that the clusters for the data rows in the dataframe are correlated, see the below aggregation of the cluster segments in the NEXT CELL BELOW...  The results show the following: Orig cluster mapping 0 --> PCA cluster 2; Orig cluster 1 --> PCA cluster 1; Orig cluster 2 --> PCA cluster 3; Orig cluster 3 --> PCA cluster 0  

# In[ ]:


predict_cluster_segments_df = pd.concat([market_data_scaled_predict_df['Cluster Segment'], market_data_scaled_pca_predict_df['Cluster Segment']], axis=1)
predict_cluster_segments_df.columns = ['Orig Data Cluster Segment', 'PCA Cluster Segment']

print("# of points in:")
for i in range(4):
    print("\t cluster " + str(i) + ":")
    print("\t\tOriginal: " + str(predict_cluster_segments_df.loc[predict_cluster_segments_df['Orig Data Cluster Segment'] == i].shape[0]))
    print("\t\tPCA:      " + str(predict_cluster_segments_df.loc[predict_cluster_segments_df['PCA Cluster Segment'] == i].shape[0]))

# Notes
# Orig cluster 0 --> PCA cluster 2
# Orig cluster 1 --> PCA cluster 1
# Orig cluster 2 --> PCA cluster 3
# Orig cluster 3 --> PCA cluster 0

print()
print("Combined ORIG + PCA Cluster Segments:")
display(predict_cluster_segments_df)


# In[ ]:




