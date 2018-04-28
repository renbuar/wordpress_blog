
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def calculate_diagnostic_performance (actual_predicted):
    """ Calculate diagnostic performance.
    
    Takes a Numpy array of 1 and zero, two columns: actual and predicted
    
    Note that some statistics are repeats with different names
    (precision = positive_predictive_value and recall = sensitivity).
    Both names are returned
    
    Returns a dictionary of results:
        
    1) accuracy: proportion of test results that are correct    
    2) sensitivity: proportion of true +ve identified
    3) specificity: proportion of true -ve identified
    4) positive likelihood: increased probability of true +ve if test +ve
    5) negative likelihood: reduced probability of true +ve if test -ve
    6) false positive rate: proportion of false +ves in true -ve patients
    7) false negative rate:  proportion of false -ves in true +ve patients
    8) positive predictive value: chance of true +ve if test +ve
    9) negative predictive value: chance of true -ve if test -ve
    10) precision = positive predictive value 
    11) recall = sensitivity
    12) f1 = (2 * precision * recall) / (precision + recall)
    13) positive rate = rate of true +ve (not strictly a performance measure)
    """
    # Calculate results
    actual_positives = actual_predicted[:, 0] == 1
    actual_negatives = actual_predicted[:, 0] == 0
    test_positives = actual_predicted[:, 1] == 1
    test_negatives = actual_predicted[:, 1] == 0
    test_correct = actual_predicted[:, 0] == actual_predicted[:, 1]
    accuracy = np.average(test_correct)
    true_positives = actual_positives & test_positives
    true_negatives = actual_negatives & test_negatives
    sensitivity = np.sum(true_positives) / np.sum(actual_positives)
    specificity = np.sum(true_negatives) / np.sum(actual_negatives)
    positive_likelihood = sensitivity / (1 - specificity)
    negative_likelihood = (1 - sensitivity) / specificity
    false_positive_rate = 1 - specificity
    false_negative_rate = 1 - sensitivity
    positive_predictive_value = np.sum(true_positives) / np.sum(test_positives)
    negative_predictive_value = np.sum(true_negatives) / np.sum(test_negatives)
    precision = positive_predictive_value
    recall = sensitivity
    f1 = (2 * precision * recall) / (precision + recall)
    positive_rate = np.mean(actual_predicted[:,1])
    
    # Add results to dictionary
    performance = {}
    performance['accuracy'] = accuracy
    performance['sensitivity'] = sensitivity
    performance['specificity'] = specificity
    performance['positive_likelihood'] = positive_likelihood
    performance['negative_likelihood'] = negative_likelihood
    performance['false_positive_rate'] = false_positive_rate
    performance['false_negative_rate'] = false_negative_rate
    performance['positive_predictive_value'] = positive_predictive_value
    performance['negative_predictive_value'] = negative_predictive_value
    performance['precision'] = precision
    performance['recall'] = recall
    performance['f1'] = f1
    performance['positive_rate'] = positive_rate

    # Return results dictionary
    return performance

def load_data():
    """Load data from appropriate source. Here we load data from scikit
    learn's buil in data sets, but we restrict the feature data to two columns
    in order to demonstrate changing sensitivity of the model"""
    
    data_set = datasets.load_breast_cancer()
    X=data_set.data[:,0:2]
    y=data_set.target
    return X, y


def normalise (X_train,X_test):
    """Normalise X data, so that training set has mean of zero and standard
    deviation of one"""
    
    # Initialise a new scaling object for normalising input data
    sc=StandardScaler() 
    # Set up the scaler just on the training set
    sc.fit(X_train)
    # Apply the scaler to the training and test sets
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)
    
    # Return normalised data
    return X_train_std, X_test_std

def set_up_k_fold_splits(k):
    """Set up K-fold splits. This will divide data into k sets such that each
    data points is used once and only once in the test set. Each test/training
    split has same balance of y (the classified outcome) as the whole data
    set"""
    
    skf = StratifiedKFold(n_splits = k)
    skf.get_n_splits(X, y)

    return skf

def test_model (model, X_test_std, y, threshold):
    y_pred_probability = model.predict_proba(X_test_std)
    # Check probability of positive classification is >trhreshold
    y_pred = (y_pred_probability[:,1] >= threshold)
    # Convert boolean to 0/1 (could also simply multiple by 1)
    y_pred = y_pred.astype(int)
    
    test_results = np.vstack((y, y_pred)).T
    
    # return results
    return test_results

def train_model (X, y):
    """Train the model """
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=100,random_state=0)
    model.fit(X, y)
    
    return model

# main model:
if __name__ == '__main__':

    # Load data
    X, y = load_data()
    
    # Set up K-fold splits. 
    splits = 10
    skf = set_up_k_fold_splits (splits)
    
    # Set up results dataframe
    cols = ['threshold',
            'k_fold',
            'accuracy',
            'sensitivity',
            'specificity',
            'positive_likelihood',
            'negative_likelihood',
            'false_positive_rate',
            'false_negative_rate',
            'positive_predictive_value',
            'negative_predictive_value',
            'precision',
            'recall',
            'f1',
            'positive_rate']
    
    results_df = pd.DataFrame(columns = cols)

    run_id = 0
    k_fold_count = 0

    for train_index, test_index in skf.split(X, y):
        
        # Divide data in train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Normalise data
        X_train_std, X_test_std = normalise(X_train,X_test)

        # Train model
        model = train_model (X_train_std,y_train)
        
        # Loop through a range of thresholds when predicting classification
        for threshold in np.arange (0.0,1.01,0.01):
            # linear regression model has .predict+proba  method to return 
            # probability of outcomes. Some methods such as svc use 
            # .decision_function to return probability
            
            # Get test results (combined with actual classification)
            test_results = test_model (model, X_test_std, y_test, threshold)
            
            # Measure performance of test set predictions
            performance = calculate_diagnostic_performance(test_results)
            run_results = [threshold, k_fold_count] + list(performance.values())
            results_df.loc[run_id] = run_results
            run_id += 1
        
        k_fold_count += 1
    
    # Summarise data
    summary_mean = results_df.groupby('threshold').mean()
    summary_stdev = results_df.groupby('threshold').std()