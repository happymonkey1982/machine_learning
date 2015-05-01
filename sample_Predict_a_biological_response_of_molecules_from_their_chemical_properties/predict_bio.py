from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV
import logloss
import numpy as np

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
        
def main():
    #read in  data, parse into training and target sets
    dataset = np.genfromtxt('data/train.csv',delimiter=',', dtype='f8')[1:]    
    target = np.array([x[0] for x in dataset])
    train = np.array([x[1:] for x in dataset])
    
    #selected based on the variance
    #sel = VarianceThreshold(threshold=(.2))
    #train = sel.fit_transform(train)
    train = SelectKBest(f_classif, k=100).fit_transform(train, target)
    
    cfr = RandomForestClassifierWithCoef(n_estimators=100)
    #Simple K-Fold cross validation. 5 folds.
    #(Note: in older scikit-learn versions the "n_folds" argument is named "k".)
    cv = cross_validation.KFold(len(train), n_folds=5)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        results.append( logloss.llfun(target[testcv], [x[1] for x in probas]) )

    #print out the mean of the cross-validated results
    print("Results: " , str( np.array(results).mean() ))
    
#    test = np.genfromtxt('data/test.csv', delimiter=',', dtype='f8')[1:]
#    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(cfr.predict_proba(test))]

#    np.savetxt('data/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
#            header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()