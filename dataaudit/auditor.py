import pandas as pd
from numpy import nan
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import json
from scipy.stats import norm

class Auditor():
    
    def write_check_violation(self, data,filepath):
        
        pretty_print = json.dumps(data, indent=4)
        with open(filepath, 'w') as outfile:
            json.dump(pretty_print, outfile, indent=4)
    
    def load_checkmeta(self, entitysets,meta_json_filepath):
        
        with open(meta_json_filepath) as file:
                  check_againset_meta = json.load(file)
                  self.create_check_list(entitysets,check_againset_meta)

    def create_check_list(self, entitysets,check_againset_meta): 
        
        #number of nans for all entities
        check_list =[]
        #creating a list of all all the check fields
        for entity in entitysets.entities:
            df =  entity.df
            entity_name = entity.id
            fields = df.columns
            for field in fields:
                for checks in check_againset_meta[entity_name]:
                    if field in checks:
                        check_list.append(checks)
                         
        violation = {entity_name: self.find_type(df,check_list)}               
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
    
        return {"min":np.float64(min(attr_value))}
     
    def find_maximum(self, attr_value):
         
        return {"max":np.float64(max(attr_value))}
        
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
        
        normal_distribution = {"normal": { "mean": np.float64(mu),"std": np.float64(std), "min":np.float64(min_value), "max":np.float64(max_value)}}
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