from cProfile import run
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from encoders import DistanceTransformer,  TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
from data import get_data, clean_data
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        # self.X = X
        # self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        # Add the model of your choice to the pipeline
        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        return self.pipe

    def run(self, X_train,y_train):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(X_train, y_train)

        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipe.predict(X_test)
        rsme = np.sqrt(((y_pred - y_test)**2).mean())
        return rsme


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # build pipeline
    trainer = Trainer()

    # train the pipeline
    trainer.run(X_train, y_train)

    # evaluate the pipeline
    rmse = trainer.evaluate(X_val, y_val)

    print(f'rmse = {rmse}')
