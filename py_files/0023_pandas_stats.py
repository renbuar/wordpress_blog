
# coding: utf-8

# # Basic statistics in Pandas
# 
# Like NumPy, Pandas may be used to give us some basic statistics on data.
# 
# Let's start by building a very sample dataframe.

# In[72]:


import pandas as pd
df = pd.DataFrame()

names = ['Gandolf','Gimli','Frodo','Legolas','Bilbo']
types = ['Wizard','Dwarf','Hobbit','Elf','Hobbit']
magic = [10, 1, 4, 6, 4]
aggression = [7, 10, 2, 5, 2]
stealth = [8, 2, 5, 10, None]


df['names'] = names
df['type'] = types
df['magic_power'] = magic
df['aggression'] = aggression
df['stealth'] = stealth


# ## Overview statistics
# 
# We can get an overview with the <em>describe()</em> method.

# In[73]:


print (df.describe())


# We can modify the percentiles reported:

# In[74]:


print (df.describe(percentiles=[0.05,0.1,0.9,0.95]))


# Specific statistics may be returned:

# In[75]:


print (df.mean())


# ## List of key statistical methods
# 
# .mean() =  mean
# 
# .median() = median
# 
# .min() = minimum
# 
# .max() =maximum
# 
# .quantile(x)
# 
# .var() = variance
# 
# .std() = standard deviation
# 
# .mad() = mean absolute variation
# 
# .skew() = skewness of distribution
# 
# .kurt() = kurtosis
# 
# .cov() = covariance
# 
# .corr() = Pearson Correlation coefficent
# 
# .autocorr() = autocorelation
# 
# .diff() = first discrete difference
# 
# .cumsum() = cummulative sum
# 
# .comprod() = cumulative product
# 
# .cummin() = cumulative minimumcs:
# 
# .mean() =  mean
# 
# .median() = median
# 
# .min() = minimum
# 
# .max() =maximum
# 
# .quantile(x)
# 
# .var() = variance
# 
# .std() = standard deviation
# 
# .mad() = mean absolute variation
# 
# .skew() = skewness of distribution
# 
# .kurt() = kurtosis
# 
# .cov() = covariance
# 
# .corr() = Pearson Correlation coefficent
# 
# .autocorr() = autocorelation
# 
# .diff() = first discrete difference
# 
# .cumsum() = cummulative sum
# 
# .comprod() = cumulative product
# 
# .cummin() = cumulative minimum

# ## Returning the index of minimum and maximum
# 
# <em>idxmin</em> and <em>idxmax</em> will return the index row of the min/max. If two values are equal the first will be returned.

# In[76]:


print ('Minimum:', df['aggression'].min())
print ('Index row:',df['aggression'].idxmin())
print ('\nFull row:\n', df.iloc[df['aggression'].idxmin()])


# ## Removing rows with incomplete data
# 
# We can extract only those rows with a complete data set using the <em>dropna()</em> method.

# In[77]:


print (df.dropna())


# We can use this directly in the describe method.

# In[78]:


print (df.dropna().describe())


# To create a new dataframe with compplete rows only, we would simply assign to a new variable name:

# In[79]:


df_na_dropped = df.dropna()

