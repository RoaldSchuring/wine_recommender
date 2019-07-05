import argparse
import pandas as pd
import os
from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors
import numpy as np
import subprocess
import sys
from sagemaker_containers.beta.framework import worker, encoders
from six import BytesIO


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('s3fs')

def _npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()

def convert_to_list(raw_review_vec):
    review_vec_trimmed = raw_review_vec.replace('[', '').replace(']', '')
    review_vec = np.fromstring(review_vec_trimmed, dtype=float, sep='  ')
    review_vec_list = review_vec.tolist()
    return review_vec_list

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def predict_fn(input_data, model):
    input_data_reshaped = input_data.reshape(1, -1)
    distance, indice = model.kneighbors(input_data_reshaped, 11)
    distance_list = distance[0].tolist()[1:]
    indice_list = indice[0].tolist()[1:]
    nearest_neighbors = [distance_list, indice_list]
    print('predict_fn output is', nearest_neighbors)
    return np.array(nearest_neighbors)

def output_fn(prediction_output, accept):
    if accept == 'application/x-npy':
        print('output_fn input is', prediction_output, 'in format', accept)
        return _npy_dumps(prediction_output), 'application/x-npy'
    elif accept == 'application/json':
        print('output_fn input is', prediction_output, 'in format', accept)
        return worker.Response(encoders.encode(prediction_output, accept), accept, mimetype=accept)
    else:
        raise ValueError('Accept header must be application/x-npy, but it is {}'.format(accept))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--metric', type=str, default='cosine')

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default='s3://data-science-wine-reviews/nearest_neighbors/output_data')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default='s3://data-science-wine-reviews/nearest_neighbors/data/wine_review_vectors.csv')

    args = parser.parse_args()

    # print('raw data has been loaded')
    raw_data = pd.read_csv(args.train)

    # raw_data = pd.read_csv(args.train)
    raw_data['review_vec'] = raw_data['review_vector'].apply(convert_to_list)
    wine_vectors_list = np.array(list(raw_data['review_vec']))
    print('data has been transformed')

    # Here we supply the hyperparameters of the nearest neighbors model
    n_neighbors = args.n_neighbors
    metric = args.metric

    # now, fit the nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model_nn = nn.fit(wine_vectors_list)
    print('model has been fitted')

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(model_nn, os.path.join(args.model_dir, "model.joblib"))
