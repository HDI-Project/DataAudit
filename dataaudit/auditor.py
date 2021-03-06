import json
import warnings
from itertools import groupby

import numpy as np
import pandas as pd
from numpy import nan
from scipy import stats
from scipy.stats import ks_2samp, kstest


class Auditor():
    '''A class responsible for data auditing.'''

    def write_check_violation(self, data, filepath):
        '''Writes all the violation into json file.

        Args:
          data: A json object, which encapsulates a new value
              for each check.
          filepath: A path for the json file.
        '''
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def load_checkmeta(self, entitysets, meta_json_filepath):
        '''Loads all the checks from the json file.

        Args:
          entitysets: The entire Fhir entitysets.
          meta_json_filepath: A path for the json file.

        Returns:
          A dictionary with resulting checks.
        '''
        with open(meta_json_filepath) as file:
            check_againset_meta = json.load(file)
            return self.create_check_list(entitysets, check_againset_meta)

    def create_check_list(self, entitysets, check_againset_meta, write=False):
        '''Creates a list of all the check fields from the json file.

        Args:
          entitysets: The entire Fhir entitysets.
          check_againset_meta: A json object, which encapsulates a value
              for each check.
          write: A flag for locally writing a json file.

        Returns:
          A dictionary with resulting checks and violations.
        '''
        # Number of nans for all entities.
        check_list = []
        violation = {}
        for entity in entitysets.entities:
            df = entity.df
            entity_name = entity.id
            if entity_name not in check_againset_meta:
                continue
            fields = df.columns
            for field in fields:
                for checks in check_againset_meta[entity_name]:
                    if field in checks:
                        check_list.append(checks)
            violation[entity_name] = self.find_type(df, check_list)
        if write:
            self.write_check_violation(violation, "violation_check_againset_meta.json")

        return violation

    def check_total_nans(self, entity_name, df):
        '''Checks the number of nans for all the attributes within a specific entity.

        Args:
            entity_name: The name of the entity.
            df: A dataframe, which encapsulates all the records of that entity.

        Returns:
            A list with the number of nans for all the attributes.
        '''
        number_of_nan = []
        dict_values = df.to_dict('list')
        attr_value_list = []
        for attr, values in dict_values.items():
            attr_value_list.append({attr: self.check_nan(values)})
        number_of_nan.append({entity_name: attr_value_list})
        return number_of_nan

    def check_nan(self, attr_value):
        '''Checks the number of nans for a specific attribute within an entity.

        Args:
            attr_value: A list of values for specific attribute.

        Returns:
            A dictionary with the number of nans for a specific attribute.
        '''
        summation = sum(
            value in [
                nan,
                'null',
                'nan',
                'NAN',
                'Nan',
                'NaN',
                'undefined',
                'unknown'] for value in attr_value)
        percentage_of_nans = (summation / len(attr_value)) * 100
        percentage_of_nans = "%.2f%%" % round(percentage_of_nans, 2)
        return {"nan": {"number": summation, "percenatge": percentage_of_nans}}

    def find_type(self, df, check_list):
        '''Finds the type of all the checks for a specific attribute.

        Args:
            df: A dataframe, which encapsulates all the records of that entity.
            check_list: A list of all the required checks for a specific attribute.

        Returns:
            A list with all the checks and their new values.
        '''
        fields_list = []
        # Extract the list of checks for each column.
        for checks in check_list:
            modefied_check_list = []
        # Get the column name.
            key = list(checks)[0]
        # Extract all values.
            attr_value = df[key].values
        # Extract the list of checks for a specific column.
            for check in checks[key]:
                if("distribution" in check):
                    modefied_check_list.append(
                        {"distribution": self.find_distribution(check["distribution"],
                                                                attr_value, 5)})
                if("min" in check):
                    modefied_check_list.append(self.find_minimum(attr_value))
                if("max" in check):
                    modefied_check_list.append(self.find_maximum(attr_value))
                if("mean" in check):
                    modefied_check_list.append(self.find_mean(attr_value))
                if("nan" in check):
                    modefied_check_list.append(self.check_nan(attr_value))
                if("freq" in check):
                    modefied_check_list.append(self.find_frequency(attr_value))
                if("percenatge" in check):
                    modefied_check_list.append(self.find_percentage(key, attr_value))
            fields_list.append({key: modefied_check_list})
        return fields_list

    def find_percentage(self, key, attr_value):
        '''Calculates the percentage frequency distribution of the attribute values.

        Args:
            key: A name of an attribute.
            attr_value: A list of values for the a specific attribute.

        Returns:
            A dictionary with the percentage for each value.
        '''
        dictionary_of_percenatges = {}
        df = pd.DataFrame({key: attr_value})
        number_of_records = len(df)
        for index, value in enumerate(df[key].value_counts()):
            attr = df[key].value_counts().keys()[index]
            percenatge = (value / number_of_records) * 100
            dictionary_of_percenatges[attr] = "%.2f%%" % round(percenatge, 2)
        return{"percenatge": dictionary_of_percenatges}

    def find_frequency(self, attr_value):
        '''Calculates the frequency of the attribute values.

        Args:
            attr_value: A list of values for the a specific attribute.

        Returns:
            A dictionary with the frequency of occurance for each value.
        '''
        freq = {key: len(list(group)) for (key, group) in groupby(attr_value)}
        return freq

    def find_minimum(self, attr_value):
        '''Finds the minimum value in a specific attribute.

        Args:
            attr_value: A list of values for the a specific attribute.

        Returns:
            A dictionary with the minimum value.
        '''
        if(all(isinstance(x, (int, float)) for x in attr_value)):
            return {"min": np.int(min(attr_value))}
        else:
            raise TypeError("The list: '{}' must be numerical".format(attr_value))

    def find_maximum(self, attr_value):
        '''Finds the maximum value in a specific attribute.

        Args:
            attr_value: A list of values for the a specific attribute.

        Returns:
            A dictionary with the maximum value.
        '''
        if(all(isinstance(x, (int, float)) for x in attr_value)):
            return {"max": np.int(max(attr_value))}
        else:
            raise TypeError("The list: '{}' must be numerical".format(attr_value))

    def find_mean(self, attr_value):
        '''Finds the mean value in a specific attribute.

        Args:
            attr_value: A list of values for the a specific attribute.

        Returns:
            A dictionary with the mean value.
        '''
        if(all(isinstance(x, (int, float)) for x in attr_value)):
            return {"mean": np.int(np.mean(attr_value))}
        else:
            raise TypeError("The list: '{}' must be numerical".format(attr_value))

    def find_distribution(self, distributions, attr_value, num):
        '''Create a list of dictionaries after comparing all the continous distributions using ks-test.

        Args:
            distributions: A list of different types of distributions for a specific attribute.
            attr_value: A list of values for the a specific attribute.

        Returns:
            A list of dictionaries after comparing all the continous distributions using ks-test.
        '''
        distributions_list = []
        all_distributions_checks = self.identify_goodness_of_fit(attr_value, num)
        distributions_list.append(all_distributions_checks)
        for distribution in distributions:
            key = list(distribution)[0]
            attr = distribution[key]
            distributions_list.append(
                self.apply_predefined_distribution_check(
                    attr_value,
                    dist_type=key,
                    dist_charecteristics=attr))
        return distributions_list

    def create_distribution_from_attr(
            self, dist_type, dist_charecteristics, n_samples=1000, seed=0):
        '''Generates a sample of values that are drawn from the chosen distribution.

        Args:
            dist_type: The name of the distribution to draw from.
            dist_charecteristics: Dictionary that contains the attributes of the distribution.
            n_samples: The number of values to draw from the distribution.

        Returns:
            A list of the sampled values from the distributions.
        '''
        np.random.seed(seed)
        if dist_type == 'normal':
            return np.random.normal(loc=dist_charecteristics['mean'],
                                    scale=dist_charecteristics['std'],
                                    size=n_samples)
        elif dist_type == 'poisson':
            return np.random.poisson(lam=dist_charecteristics['lambda'], size=n_samples)
        elif dist_type == 'beta':
            return np.random.beta(a=dist_charecteristics['a'],
                                  b=dist_charecteristics['b'],
                                  size=n_samples)
        elif dist_type == 'gamma':
            return np.random.gamma(shape=dist_charecteristics['k'],
                                   scale=dist_charecteristics['theta'],
                                   size=n_samples)
        elif dist_type == 'weibull':
            return np.random.weibull(a=dist_charecteristics['a'], size=n_samples)
        elif dist_type == 'uniform':
            return np.random.uniform(low=dist_charecteristics['low'],
                                     high=dist_charecteristics['high'],
                                     size=n_samples)
        elif dist_type == 'powerlaw':
            return np.random.power(a=dist_charecteristics['a'], size=n_samples)
        elif dist_type == 'expon':
            return np.random.exponential(scale=dist_charecteristics['scale'], size=n_samples)

    def compare_distributions_one_sample(self, x, dist_type, args=()):
        '''Compares one sample of values to a certain distribution.

        Args:
            x: A 1-D array of observations of random variable.
            dist_type: The name of the distribution to compare to.
            args: The arguments of the distribution according to scipy continous distributions.

        Returns:
            A dictionary of the comparison result using ks-test
            ({'statistic': 0.1, 'pvalue': 0.01}).
        '''
        result = {}
        ks_result = kstest(x, dist_type, args=args)
        result['ks_test'] = {'statistic': ks_result[0], 'pvalue': ks_result[1]}
        return result

    def compare_distributions(self, x, y):
        '''Compares if two samples come from the same distribution.

        Args:
            x: A 1-D array of observations of random variable.
            y: A 1-D array of observations of random variable.
        Returns:
            A dictionary of the comparison result using ks-test
            ({'statistic': 0.1, 'pvalue': 0.01}).
        '''
        result = {}
        ks_result = ks_2samp(x, y)
        result['ks_test'] = {'statistic': ks_result[0], 'pvalue': ks_result[1]}
        return result

    def identify_goodness_of_fit(self, x, num):
        '''Generates the most fitted distribution to the given observations.

        Args:
            x: A 1-D array of observations of random variable.

        Returns:
            A dictionary of the comparison result using ks-test for all continous distributions.
        '''
        list_of_distributions = [stats.alpha, stats.anglit, stats.arcsine, stats.beta,
                                 stats.betaprime, stats.bradford, stats.wald,
                                 stats.burr, stats.cauchy, stats.chi, stats.chi2,
                                 stats.cosine, stats.dgamma, stats.dweibull,
                                 stats.erlang, stats.expon, stats.exponnorm,
                                 stats.exponweib, stats.exponpow, stats.f,
                                 stats.fatiguelife, stats.fisk, stats.foldcauchy,
                                 stats.foldnorm, stats.frechet_r, stats.lognorm,
                                 stats.frechet_l, stats.genlogistic, stats.genpareto,
                                 stats.gennorm, stats.genexpon, stats.genextreme,
                                 stats.gausshyper, stats.gamma, stats.gengamma,
                                 stats.genhalflogistic, stats.gilbrat, stats.gompertz,
                                 stats.gumbel_r, stats.gumbel_l, stats.halfcauchy,
                                 stats.halflogistic, stats.halfnorm, stats.halfgennorm,
                                 stats.hypsecant, stats.invgamma, stats.invgauss,
                                 stats.invweibull, stats.johnsonsb, stats.johnsonsu,
                                 stats.kstwobign, stats.laplace, stats.levy, stats.levy_l,
                                 stats.logistic, stats.loggamma, stats.loglaplace,
                                 stats.lomax, stats.maxwell, stats.mielke, stats.nakagami,
                                 stats.ncx2, stats.ncf, stats.nct, stats.norm, stats.pareto,
                                 stats.pearson3, stats.powerlaw, stats.powerlognorm,
                                 stats.powernorm, stats.rdist, stats.reciprocal,
                                 stats.rayleigh, stats.rice, stats.recipinvgauss,
                                 stats.semicircular, stats.t, stats.triang, stats.ksone,
                                 stats.truncexpon, stats.truncnorm, stats.tukeylambda,
                                 stats.uniform, stats.vonmises, stats.vonmises_line,
                                 stats.weibull_min, stats.weibull_max, stats.wrapcauchy]
        list_of_distributions = list_of_distributions[:num]
        ks_for_all_distributions = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for dist in list_of_distributions:
                mle_result = dist.fit(x)
                result = self.compare_distributions_one_sample(x, dist.name, args=mle_result)
                ks_for_all_distributions.append({'name': dist.name,
                                                 'ks statistic': result['ks_test']['statistic'],
                                                 'ks pvalue': result['ks_test']['pvalue'],
                                                 'mle args': list(mle_result)})
        return {'most_fit_distribution': ks_for_all_distributions}

    def apply_predefined_distribution_check(self, x, dist_type='normal', dist_charecteristics={}):
        '''Applies the one and two samples distribution comparisons.

        Args:
            x: A 1-D array of observations of random variable.
            dist_type: The name of the distribution to draw from.
            dist_charecteristics: A dictionary that contains the attributes of the distribution.

        Returns:
            A dictionary of the comparison result using ks-test for specific continous
            distributions.
        '''
        simulated_dist = self.create_distribution_from_attr(
            dist_type=dist_type, dist_charecteristics=dist_charecteristics, n_samples=10000)
        if simulated_dist is not None:
            simulated_dist_result = self.compare_distributions(x, simulated_dist)
        else:
            simulated_dist_result = {
                'ks_test': 'Incomplete distribution information\
                 or distribution error not available'}
        return {dist_type: simulated_dist_result}
