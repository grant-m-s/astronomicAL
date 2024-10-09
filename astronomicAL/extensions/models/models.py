from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .. import models

model_name = sorted(name for name in models.__dict__)


def get_classifiers():
    classifiers_nets = {
        m: models.__dict__[m] for m in model_name
    }
    classifiers = {
        "KNN": KNeighborsClassifier(3, n_jobs=-1),
        "DTree": DecisionTreeClassifier(
            random_state=0,
        ),
        "RForest": RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=1000),
        "AdaBoost": AdaBoostClassifier(random_state=0, n_estimators=500),
        "GBTrees": GradientBoostingClassifier(random_state=0, n_estimators=1000),
    }

    for i,j in classifiers_nets.items():
        if "__" in i:
            continue
        classifiers[i] = j
    return classifiers
