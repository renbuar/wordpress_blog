# See https://docs.scipy.org/doc/scipy/reference/stats.html for dists



import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# For full list of possible distributions to tets see 
# https://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html#module-scipy.stats



size = 1000

# creating the dummy sample (using beta distribution)
y = np.random.lognormal(3, 1, size)

y = pd.read_csv('my_data.csv').values

x = scipy.arange(len(y))



#y = np.linspace(-10,10,10)




dist_names = ['beta',
              'expon',
              'gamma',
              'lognorm',
              'norm',
              'pearson3',
              'triang',
              'uniform',
              'weibull_min', 
              'weibull_max']



p_values = []
chi_square = []
percentile_bins = np.linspace(0,100,51)

sc=StandardScaler() 
yy = y.reshape (-1,1)
sc.fit(yy)
y_std =sc.transform(yy)
y_std = y_std.flatten()

percentile_cutoffs = np.percentile(y_std, percentile_bins)

for distribution in dist_names:
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = int((p*1e4)+0.5)/1e4
    p_values.append(p)
    # Get expected counts in percentile bins
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
    
    expected_frequency = []
    for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
    # calculate sum of squares of observed - expected
    expected_frequency = np.array(expected_frequency)
    ss = sum (((expected_frequency - 0.02) ** 2) / 0.02)
    chi_square.append(ss)
        
    
    
print ('\nDistributions sorted by goodness of fit:')
print ('----------------------------------------')
results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results.sort_values(['chi_square'], inplace=True)

print (results)

# Plot best distributions

number_of_bins = 100
bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)


number_distributions_to_plot = 1

print ('\nBest distributions:')
print ('-------------------')

h = plt.hist(y, bins = bin_cutoffs, color='0.75')

dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]
parameters = []


for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    starting_loc = np.mean(y)

    if len(x) > 30000000:
        np.random.shuffle(y)
        sample = y[0:100]
        starting_loc = dist.fit(sample)[0]
    param = dist.fit(y, starting_loc)
    parameters.append(param)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
    pdf_fitted *= scale_pdf
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,np.percentile(y,98))


plt.legend()
plt.show()

dist_parameters = pd.DataFrame()
dist_parameters['Distribution'] = (
        results['Distribution'].iloc[0:number_distributions_to_plot])
dist_parameters['Distribution parameters'] = parameters

print ('\nDistribution parameters:')
print ('------------------------')

for index, row in dist_parameters.iterrows():
    print ('\nDistribution:', row[0])
    print ('Parameters:', row[1] )
    



    
    
        