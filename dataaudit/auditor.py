import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import json
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

class Auditor():
    
    def write_check_violation(self, data,filepath):
        
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    
    def load_checkmeta(self, entitysets,meta_json_filepath):
        
        with open(meta_json_filepath) as file:
                  check_againset_meta = json.load(file)
                  self.create_check_list(entitysets,check_againset_meta)

    def create_check_list(self, entitysets,check_againset_meta): 

        #number of nans for all entities
        check_list =[]
        violation ={}
        #creating a list of all all the check fields
        for entity in entitysets.entities:
            df =  entity.df
            entity_name = entity.id
            if entity_name not in check_againset_meta:
                continue
            fields = df.columns
            for field in fields:
                for checks in check_againset_meta[entity_name]:
                    if field in checks:
                        check_list.append(checks)
            violation[entity_name] = self.find_type(df,check_list)                 
        self.write_check_violation(violation , "violation_check_againset_meta.json")
        
    
    def check_total_nans(self,entity_name,df):
        
        #checks the number of nans for all the columns within a specific entity 
        number_of_nan =[]
        dict_values = df.to_dict('list')
        attr_value_list = []
        for attr, values in dict_values.items():
                        summation = sum(value in [nan,'null','nan','NAN','Nan','NaN', 'undefined', 'unknown'] for value in values)
                        percentage_of_nans = (summation/len(values))*100
                        percentage_of_nans = "%.2f%%" % round(percentage_of_nans,2)
                        attr_value_list.append({attr:[summation,percentage_of_nans]})
        number_of_nan.append({entity_name:attr_value_list})
        return number_of_nan
    
    def check_nan(self,attr_value):
        
        #number of nans for a specific entity
        summation = sum(value in [nan,'null','nan','NAN','Nan','NaN', 'undefined', 'unknown'] for value in attr_value)
        percentage_of_nans = (summation/len(attr_value))*100
        percentage_of_nans = "%.2f%%" % round(percentage_of_nans,2)
        return {"nan": {"number":summation,"percenatge":percentage_of_nans}}
    
    def find_type(self,df,check_list):
        
        fields_list = []
        #extract the list of checks for each column
        for checks in check_list:
            modefied_check_list = []
        # get the column name 
            key = __builtins__.list(checks)[0]
        #extract all values
            attr_value = df[key].values
        #extract the list of checks for a specific column
            for check in checks[key]:
                if("distribution" in check):
                    modefied_check_list.append({"distribution":self.find_distribution(check["distribution"],attr_value)})
                if("min" in check):
                    modefied_check_list.append(self.find_minimum(attr_value))
                if("max" in check):
                    modefied_check_list.append(self.find_maximum(attr_value))
                if("nan" in check):
                    modefied_check_list.append(self.check_nan(attr_value))
                if("percenatge" in check):
                    modefied_check_list.append(self.find_percentage(key,attr_value))        
            fields_list.append({key:modefied_check_list})     
        return fields_list
    
    
    
    def find_percentage(self,key,attr_value):
        
        dictionary_of_percenatges ={}
        df = pd.DataFrame({key:attr_value})
        number_of_records= len(df)
        for index,value in enumerate(df[key].value_counts()):
            attr = df[key].value_counts().keys()[index]
            percenatge = (value/number_of_records)*100
            dictionary_of_percenatges[attr] = "%.2f%%" % round(percenatge,2)
        return{"percenatge":dictionary_of_percenatges}
                    
    def find_minimum(self, attr_value):
    
        return {"min":np.int(min(attr_value))}
     
    def find_maximum(self, attr_value):
         
        return {"max":np.int(max(attr_value))}
        
    def find_distribution(self, distributions, attr_value):
        
        distributions_list =[]
        for distribution in distributions:
            key = __builtins__.list(distribution)[0]
            attr = distribution[key]
            if(key == 'normal'):
                distributions_list.append(self.normal_distribution(attr_value, attr['min'],attr['max'],attr['mean'],attr['std']))
                
        return distributions_list
    
    
    def normal_distribution(self, attr_value,min_x, max_x, mu, sigma):
    
        # Expected normal distribution
        x = np.linspace(min_x, max_x)
        plt.plot(x,mlab.normpdf(x, mu, sigma), linestyle='dashed')
    
        # Fit a normal distribution to the data:
        binwidth = 1
        min_value = min(attr_value)
        max_value = max(attr_value)
        bins = range(min_value, max_value + binwidth, binwidth)
    
        mu, std = norm.fit(attr_value)
        
        normal_distribution = {"normal": { "mean": mu,"std": std, "min":np.int(min_value), "max":np.int(max_value)}}
        # Plot the histogram.
        plt.figure()
        plt.hist(attr_value, bins=bins, density=True, alpha=0.6, color='g')
    
        # Plot the PDF.
        plt.figure()
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        plt.show()
        
        return normal_distribution

def create_distribution_from_attr(dist_type ='normal',dist_charecteristics = None,n_samples = 1000):

    if dist_type == 'normal':
        return np.random.normal(loc=dist_charecteristics['mu'], scale=dist_charecteristics['std'], size=n_samples)
    elif dist_type == 'poisson':
        return np.random.poisson(lam=dist_charecteristics['lambda'], size=n_samples)
    elif dist_type == 'beta':
        return np.random.beta(a=dist_charecteristics['a'],b=dist_charecteristics['b'], size=n_samples)
    elif dist_type == 'gamma':
        return np.random.gamma(shape=dist_charecteristics['k'], scale=dist_charecteristics['theta'], size=n_samples)
    elif dist_type == 'weibull':
        return np.random.weibull(a=dist_charecteristics['a'], size=n_samples)
    
def compare_distributions(x,y):
    result = {}
    
    ttest_1samp_res = ttest_1samp(x,np.mean(y))
    result['ttest_1samp_res'] = ttest_1samp_res
    
    ttest_ind_res = ttest_ind(x,y)
    result['ttest_ind_res'] = ttest_ind_res
    
    ttest_rel_res = ttest_rel(x,y)
    result['ttest_rel_res'] = ttest_rel_res
    
    x_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)
    x_bins = np.linspace(min(x),max(x))
    x_log_dens = x_kde.score_samples(x_bins.reshape(-1,1))
    
    y_kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(y)
    y_bins = np.linspace(min(y),max(y))
    y_log_dens = y_kde.score_samples(y_bins.reshape(-1,1))
    
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    
    ax.plot(x_bins,x_log_dens)
    ax.plot(y_bins,y_log_dens)
    
    
    
    return result
