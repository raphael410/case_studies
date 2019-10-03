from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        print(self)
        self.model = make_pipeline(StandardScaler(), 
                                   #LogisticRegression(), 
                                   MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2)),
                                  )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def __str__(self):
        return "Classifier object, version 1.3"