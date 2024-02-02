"""Build predictive models."""
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix

class RandomForests_SFS:

    """Build a RF predictor that takes the SFS as input."""

    def __init__(self, config, sfs, labels):
        self.config = config
        self.sfs = sfs
        self.labels = labels
        self.rng = np.random.default_rng(self.config['seed'])

    def build_rf_sfs(self):

        labels = [item for sublist in self.labels for item in sublist]

        train_test_seed = self.rng.integers(2**32, size=1)[0]

        X_train, X_test, y_train, y_test = train_test_split(self.sfs, labels, test_size=0.2, random_state=train_test_seed)
        print(X_train.shape)

        sfs_rf = RandomForestClassifier(n_estimators=100, oob_score=True)

        sfs_rf.fit(X_train, y_train)
        print("Out-of-Bag (OOB) Error:", 1.0 - sfs_rf.oob_score_)


        cv_scores = cross_val_score(sfs_rf, X_test, y_test, cv=2)
        print("Cross-validation scores:", cv_scores)

        y_pred_cv = cross_val_predict(sfs_rf, X_test, y_test, cv=2)
        conf_matrix = confusion_matrix(y_test, y_pred_cv)
        print("Confusion Matrix:")
        print(conf_matrix)
