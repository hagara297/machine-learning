import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import (
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice, 
    semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
    )

distributions = [
    norm, beta, expon, gamma, genextreme, logistic, lognorm, triang, uniform, fatiguelife,            
    gengamma, gennorm, dweibull, dgamma, gumbel_r, powernorm, rayleigh, weibull_max, weibull_min, 
    laplace, alpha, genexpon, bradford, betaprime, burr, fisk, genpareto, hypsecant, 
    halfnorm, halflogistic, invgauss, invgamma, levy, loglaplace, loggamma, maxwell, 
    mielke, ncx2, ncf, nct, nakagami, pareto, lomax, powerlognorm, powerlaw, rice, 
    semicircular, trapezoid, rice, invweibull, foldnorm, foldcauchy, cosine, exponpow, 
    exponweib, wald, wrapcauchy, truncexpon, truncnorm, t, rdist
    ]
#from sklearn.model_selection import train_test_split

data_set = pd.read_csv("D:\\Desktop\\GUC Sem 4\\data.csv")
#classifying the columns into int , float ,categorical
data_set_information= data_set.info()
# prints the total sum of null values(NAN) in each columnn
#print(data_set.isnull().sum())
fill_nan = data_set.interpolate(method='linear')
#point no.3  returns the rows with null values
nan_data = data_set[data_set.isna().any(axis=1)]
#no.4  meaning we have no infinite fileds as it returns NaN print(infinte_dataset)#will return 0 rows that contains infinte fields meaning there are no inbfinte fields
infinte_dataset = fill_nan[fill_nan.isin([np.inf, -np.inf]).any(axis=1)]

#calculating the mean and the variance 
mean_per_column = fill_nan.mean(axis=0)
variance_per_column = fill_nan.var(axis=0)

#no.6 ---->plotting histogram

# fill_nan.hist(bins=30)
# plt.xlim([0,100])
# plt.ylim([50,500])
# plt.title("Data")
# plt.show()

#-------> categorizing
def columns_categories(data_set):
    object_columns = []
    float_columns = []
    int_columns = []
    other_columns = []
    n,m,s=0,0,0
    for i in data_set.columns.values:
        if data_set[i].dtypes=='object':
          object_columns.append(i)
          n+=1
        if data_set[i].dtypes=='int64':
          int_columns.append(i)
          m+=1
        if data_set[i].dtypes=='float':
          float_columns.append(i)
          s+=1
   # print('object(',n,'):\n',object_columns)
    #print('int(',m,'):\n',int_columns)
    #print('float(',s,'):\n',float_columns)
    return {"object_columns":object_columns, "float_columns":float_columns, "int_columns":int_columns}

before_fill=columns_categories(data_set)
after_fill=columns_categories(fill_nan)


# #---->plotting pdf for continuous data field
# for name in after_fill["float_columns"]:
#     fill_nan[name].plot.density()
#     plt.show()

# #----> plotting pmf for discrete data fields

# for name in after_fill["int_columns"]:
#     pmf= fill_nan[name].value_counts().sort_index()/len(fill_nan[name])
#     pmf.plot(kind="bar")
#     plt.show()

# for name in after_fill["object_columns"]:
#     pmf = fill_nan[name].value_counts().sort_index()/len(fill_nan[name])
#     pmf.plot(kind="bar")
#     plt.show()


# #---->removing the unchurnning customers

x_copy=fill_nan.copy()
x_copy.drop(x_copy[x_copy.churn==0].index,inplace=True)
#print(x_copy.churn)

# # #plotting pmf before and after removal of churning customers
# for name in after_fill["int_columns"]:
#     pmf= x_copy[name].value_counts().sort_index()/len(x_copy[name])
#     pmf.plot(kind="bar")
#     plt.show()
    
# a_pmf= x_copy.churn.value_counts().sort_index()/len(x_copy.churn)
# a_pmf.plot(kind="bar")
# plt.show()

#  #---->plotting pdf for continuous data field
# for name in after_fill["float_columns"]:
#      x_copy[name].plot.density()
#      plt.show()

# for name in after_fill["object_columns"]:
#      pmf = x_copy[name].value_counts().sort_index()/len(x_copy[name])
#      pmf.plot(kind="bar")
#      plt.show()



# #------> calculating the covariance of each field

# hagar=print(fill_nan.cov())


#-----> calculating the 2-D probability bet. any 2 variables
#-----> point 11 joint probability
# z = fill_nan.groupby(['churn', 'creditcd']).size()/len(fill_nan)
# print(z)


training_data=fill_nan.sample(frac=0.8 , random_state=25)
print(training_data)
testing_data=fill_nan.drop(training_data.index)
print(testing_data)


q=fill_nan.select_dtypes(include=[training_data.int])






#print(fill_nan.groupby('hnd_webcap')['churn'].value_counts(normalize=True).unstack(fill_value=0))
# col1=fill_nan["churn"]
# col2=fill_nan["hnd_webcap"]
# x=fill_nan["churn"].unique()
# y=fill_nan["hnd_webcap"].unique()
# length = len(fill_nan)
# for xunique in x:
#     for yunique in y:
#         p=len(fill_nan.where(col1==xunique)&(col2==yunique))/len(data_set)
#         print (p)
