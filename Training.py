from __future__ import print_function
import numpy as np
import argparse
import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.externals import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    # Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    # Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    # Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "wine.csv")
    dataset = pd.read_csv(file, engine="python")

    # Assign X and y for white wines only

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the data into Training Set and Test Set

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rfObj = RandomForestClassifier(n_estimators=130, criterion='entropy', bootstrap=False)
    rfObj.fit(X_train, y_train)
    joblib.dump(rfObj, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    rfObj = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rfObj