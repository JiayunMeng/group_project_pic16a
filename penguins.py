import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

def prep_data(df):    
    s = ['studyName', 'Sample Number', 'Region', 'Stage', 'Individual ID', 'Date Egg', 'Comments']
    penguins = df.drop(s, axis=1)
    # drop irrelevant or useless columns
    
    penguins = penguins.dropna()
    # drop samples containing NaN value
    
    penguins = penguins[penguins['Sex'] != '.']
    # drop a sample whose sex value is peculiar
    
    penguins['Species'] = penguins['Species'].str.split().str.get(0)
    # pick the first word in the name of species as an abbreviate
    
    le = preprocessing.LabelEncoder()
    penguins['Clutch Completion'] = le.fit_transform(penguins['Clutch Completion'])
    penguins['Sex'] = le.fit_transform(penguins['Sex'])
    penguins['Island'] = le.fit_transform(penguins['Island'])
    penguins['Species'] = le.fit_transform(penguins['Species'])
    # change all qualitative data to quantative data
    
    return penguins

    ''' Return the cleaned data frame
    Args: 
        df: A data frame
    Returns: 
        penguins: A data frame with useless columns dropped, redundant column content shortened and quanlitative columns digitized
    '''

class pred_alg_comb:
    def __init__(self, X_train, y_train, X_test, y_test, alg_type, comb):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        if alg_type not in ['rf', 'gb']:
            raise ValueError('The algorithm should be either Random Forest or Gradient Boosting.')
        else:
            self.alg_type = alg_type
        self.comb = comb
        ''' Initialize dictionary
        Args:
            X_train: a data frame, training set of predictor features
            y_train: a data frame, traning set of the target feature 
            X_test: a data frame, testing set of predictor features
            y_test: a data frame, testing set of the target feature 
            alg_type: a string, indicating the type of algorithm used to predict
            comb: a list with tuples as elements, the tuples are combinations of predictor features
        Returns:
            None
        '''
    
    def model(self, f):
        if len(f) != 3:
            raise ValueError('Three predictor features should be entered.')
        for feature in f:
            if feature not in self.X_train.columns:
                raise ValueError('At least one of the features entered is not a predictor feature.')
        # check the input are three predictor features 
        
        if self.alg_type == 'rf':
            alg = RandomForestClassifier(n_estimators = 40)
        elif self.alg_type == 'gb':
            alg = GradientBoostingClassifier(n_estimators = 40)
            
        alg.fit(self.X_train[[f[0], f[1], f[2]]], self.y_train)
        cv = cross_val_score(alg, self.X_train[[f[0], f[1], f[2]]], self.y_train, cv=10)
        score = cv.mean()
        return alg, score
        ''' Get and score the model trained by certain predictor features
        Args:
            f: a tuple, some predictor features
        Returns:
            alg: a classifier, the trained model 
            score: a float, the mean cross validation score of the model
        '''
    
    def find_features(self):
        d = {c:self.model(c) for c in self.comb}
        d_score = {c:d[c][1] for c in d.keys()} # make a dictionary with combination as key and the cross validation score of the corresponding model as value
        self.best_comb = max(d_score, key=d_score.get)
        self.best_model = d[self.best_comb][0]
        return self.best_comb, self.best_model

    ''' Find the best combination of predictor features
    Args:
        None
    Returns:
        self.best_comb: a tuple, the combination of predictor features that has the highest mean cross validation score
        self.best_model: a classifier, the model trained by the best combination 
    ''' 
    
    def score_test(self):
        self.score = self.best_model.score(self.X_test[[self.best_comb[0], self.best_comb[1], self.best_comb[2]]], self.y_test)
        return self.score
    ''' Find how well the best model trained by the best combination of predictor features predicts the target feature
    Args:
        None
    Returns:
        self.score: a float, indicating the accuracy of the prediction by the best model
    '''

def plot_graph(df, col, t):
    fig, ax = plt.subplots(len(col), figsize=(20,50))
    i = 0
    for c in col:
        ax[i].scatter(df[c], df[t])
        ax[i].set(xlabel = c, ylabel = t)
        i = i+1
    ''' Plots showing the relationship between some predictor features and the target feature
    Args:
        df: a dataframe, containing all features
        col: a list, names of predictor features
        t: a string, name of the target feature
    Returns:
        None
    '''