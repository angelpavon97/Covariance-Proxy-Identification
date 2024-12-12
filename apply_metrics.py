from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
import numpy as np

def fair_metrics(df, protected_attribute = 'gender', favorable_class = 1, privileged_class = 0):

    y_pred = df['y_pred']
    df = df.drop(['y_pred'], axis=1)

    dataset = StandardDataset(df, 
                          label_name='Class', 
                          favorable_classes=[favorable_class], 
                          protected_attribute_names=[protected_attribute], 
                          privileged_classes=[[privileged_class]])

    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred

    attr = dataset_pred.protected_attribute_names[0]
    idx = dataset_pred.protected_attribute_names.index(attr)
    
    privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
    unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    # metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    results = {'statistical_parity_difference': statistical_parity_difference(classified_metric),
             'predictive_parity_difference': predictive_parity_difference(classified_metric),
             'false_positive_error_rate_balance': false_positive_error_rate_balance(classified_metric),
             'equal_opportunity_difference': equal_opportunity_difference(classified_metric),
             'false_negative_rate_difference': false_negative_rate_difference(classified_metric),
             'equalized_odds': equalized_odds(classified_metric),
             'overall_accuracy_equality': overall_accuracy_equality(classified_metric)}

    # results = {'statistical_parity_difference': statistical_parity_difference(classified_metric),
    #         'predictive_parity_difference': predictive_parity_difference(classified_metric),
    #         'false_positive_error_rate_balance': false_positive_error_rate_balance(classified_metric),
    #         'equal_opportunity_difference': equal_opportunity_difference(classified_metric),
    #         'false_negative_rate_difference': false_negative_rate_difference(classified_metric),
    #         'equalized_odds': equalized_odds(classified_metric),
    #         'conditional_use_accuracy_equality': conditional_use_accuracy_equality(classified_metric),
    #         'overall_accuracy_equality': overall_accuracy_equality(classified_metric),
    #         'treatment_equality_difference': treatment_equality(classified_metric)}
        
    return results

def statistical_parity_difference(classified_metric):
    return classified_metric.statistical_parity_difference()

def predictive_parity_difference(classified_metric):
    return classified_metric.positive_predictive_value(privileged=False) - classified_metric.positive_predictive_value(privileged=True)

def false_positive_error_rate_balance(classified_metric):
    return classified_metric.false_positive_rate_difference()

def equal_opportunity_difference(classified_metric):
    return classified_metric.equal_opportunity_difference()

def false_negative_rate_difference(classified_metric): # Equivalent to equal opportunity difference
    return classified_metric.false_negative_rate_difference()

def equalized_odds(classified_metric):
    return classified_metric.average_abs_odds_difference()

def overall_accuracy_equality(classified_metric):
    return classified_metric.accuracy(privileged=False) - classified_metric.accuracy(privileged=True)

def conditional_use_accuracy_equality(classified_metric):
    PPV_difference = classified_metric.positive_predictive_value(privileged=False) - classified_metric.positive_predictive_value(privileged=True)
    NPV_difference = classified_metric.negative_predictive_value(privileged=False) - classified_metric.negative_predictive_value(privileged=True)

    return (abs(PPV_difference) + abs(NPV_difference))/2

def treatment_equality(classified_metric):

    ratio_privileged = classified_metric.num_false_negatives(privileged=True) / classified_metric.num_false_positives(privileged=True)
    ratio_unprivileged = classified_metric.num_false_negatives(privileged=False) / classified_metric.num_false_positives(privileged=False)

    return ratio_unprivileged - ratio_privileged

# Similarity-based metrics

def causal_discrimination(X, clf, protected_attribute = 'gender'):
    X_opposite = X.copy()
    X_opposite[protected_attribute] = X_opposite[protected_attribute].replace({1:0, 0:1})
    
    y = clf.predict(X)
    y_opposite = clf.predict(X_opposite)

    # get percentage of different predictions 
    res = [y[i] == y_opposite[i] for i in range(len(y))].count(False) / len(y)
    
    return res