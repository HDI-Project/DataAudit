import json
import warnings

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import nan
from scipy import stats
from scipy.stats import ks_2samp, kstest, norm


class Auditor():

    def write_check_violation(self, data, filepath):

        with open(filepath, 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def load_checkmeta(self, entitysets, meta_json_filepath):

        with open(meta_json_filepath) as file:
            check_againset_meta = json.load(file)
            self.create_check_list(entitysets, check_againset_meta)

    def create_check_list(self, entitysets, check_againset_meta):

        # Number of nans for all entities.
        check_list = []
        violation = {}
        # Creating a list of all all the check fields.
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
        self.write_check_violation(violation, "violation_check_againset_meta.json")

    def check_total_nans(self, entity_name, df):

        # Checks the number of nans for all the columns within a specific entity.
        number_of_nan = []
        dict_values = df.to_dict('list')
        attr_value_list = []
        for attr, values in dict_values.items():
            summation = sum(
                value in [
                    nan,
                    'null',
                    'nan',
                    'NAN',
                    'Nan',
                    'NaN',
                    'undefined',
                    'unknown'] for value in values)
            percentage_of_nans = (summation / len(values)) * 100
            percentage_of_nans = "%.2f%%" % round(percentage_of_nans, 2)
            attr_value_list.append({attr: [summation, percentage_of_nans]})
        number_of_nan.append({entity_name: attr_value_list})
        return number_of_nan

    def check_nan(self, attr_value):

        # Number of nans for a specific entity.
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

        fields_list = []
        # Extract the list of checks for each column.
        for checks in check_list:
            modefied_check_list = []
        # Get the column name.
            key = __builtins__.list(checks)[0]
        # Extract all values.
            attr_value = df[key].values
        # Extract the list of checks for a specific column.
            for check in checks[key]:
                if("distribution" in check):
                    modefied_check_list.append(
                        {"distribution": self.find_distribution(check["distribution"],
                                                                attr_value)})
                if("min" in check):
                    modefied_check_list.append(self.find_minimum(attr_value))
                if("max" in check):
                    modefied_check_list.append(self.find_maximum(attr_value))
                if("nan" in check):
                    modefied_check_list.append(self.check_nan(attr_value))
                if("percenatge" in check):
                    modefied_check_list.append(self.find_percentage(key, attr_value))
            fields_list.append({key: modefied_check_list})
        return fields_list

    def find_percentage(self, key, attr_value):

        dictionary_of_percenatges = {}
        df = pd.DataFrame({key: attr_value})
        number_of_records = len(df)
        for index, value in enumerate(df[key].value_counts()):
            attr = df[key].value_counts().keys()[index]
            percenatge = (value / number_of_records) * 100
            dictionary_of_percenatges[attr] = "%.2f%%" % round(percenatge, 2)
        return{"percenatge": dictionary_of_percenatges}

    def find_minimum(self, attr_value):

        return {"min": np.int(min(attr_value))}

    def find_maximum(self, attr_value):

        return {"max": np.int(max(attr_value))}

    def find_distribution(self, distributions, attr_value):

        distributions_list = []
        for distribution in distributions:
            key = __builtins__.list(distribution)[0]
            attr = distribution[key]
            distributions_list.append(
                self.apply_distribution_check(
                    attr_value,
                    dist_type=key,
                    dist_charecteristics=attr))
        return distributions_list

    def normal_distribution(self, attr_value, min_x, max_x, mu, sigma):

        # Expected normal distribution.
        x = np.linspace(min_x, max_x)
        plt.plot(x, mlab.normpdf(x, mu, sigma), linestyle='dashed')

        # Fit a normal distribution to the data.
        binwidth = 1
        min_value = min(attr_value)
        max_value = max(attr_value)
        bins = range(min_value, max_value + binwidth, binwidth)

        mu, std = norm.fit(attr_value)

        normal_distribution = {
            "normal": {
                "mean": mu,
                "std": std,
                "min": np.int(min_value),
                "max": np.int(max_value)}}
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

    def create_distribution_from_attr(
            self, dist_type, dist_charecteristics, n_samples=1000):
        '''Generate a sample of values that are drawn from the chosen distribution.

        Args:
            dist_type: str
                The name of the distribution to draw from.
            dist_charecteristics: dict
                dictionary that contains the attributes of the distribution.
            n_samples: int,optional
                the number of values to draw from the distribution

        Returns:
            A list of the sampled values from the distributions.
        '''
        try:
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
        except KeyError as e:
            print(e.args)
            return None

    def compare_distributions_one_sample(self, x, dist_type, args=()):
        '''compares one sample of values to a certain distribution.
        Args:
            x: array
                1-D array of observations of random variable.
            dist_type: dict
                The name of the distribution to compare to.
            args: tuple
                the arguments of the distribution according to scipy continous distributions

        Returns:
            A dictionary of the comparison result using ks-test ({'statistic': 0.1, 'pvalue': 0.01}).
        '''
        result = {}
        ks_result = kstest(x, dist_type, args=args)
        result['ks_test'] = {'statistic': ks_result[0], 'pvalue': ks_result[1]}
        return result

    def compare_distributions(self, x, y):
        '''compares if two samples come from the same distribution.
        Args:
            x: array
                1-D array of observations of random variable.
            y: array
                1-D array of observations of random variable.
        Returns:
            A dictionary of the comparison result using ks-test ({'statistic': 0.1, 'pvalue': 0.01}).
        '''
        result = {}
        ks_result = ks_2samp(x, y)
        result['ks_test'] = {'statistic': ks_result[0], 'pvalue': ks_result[1]}
        return result

    def identify_goodness_of_fit(self, x):
        '''Generate the most fitted distribution to the given observations
        Args:
            x: array
                1-D array of observations of random variable.
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
                break
        return ks_for_all_distributions

    def apply_distribution_check(self, x, dist_type='normal', dist_charecteristics={}):
        '''apply the one and two samples distribution comparisons
        Args:
            x: array
                1-D array of observations of random variable.
            dist_type: str,optional
                The name of the distribution to draw from.
            dist_charecteristics: dict,optional
                dictionary that contains the attributes of the distribution.
        Returns:
            A dictionary of the comparison result using ks-test for all continous distributions.
        '''
        simulated_dist = self.create_distribution_from_attr(
            dist_type=dist_type, dist_charecteristics=dist_charecteristics, n_samples=10000)
        if simulated_dist is not None:
            simulated_dist_result = self.compare_distributions(x, simulated_dist)
        else:
            simulated_dist_result = {
                'ks_test': 'Incomplete distribution information\
                 or distribution error not available'}
        most_fit_dist = self.identify_goodness_of_fit(x)
        simulated_dist_result['most_fit_distribution'] = most_fit_dist
        return {dist_type: simulated_dist_result}
