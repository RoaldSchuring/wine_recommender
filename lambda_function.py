import json
import boto3
import pandas as pd
import numpy as np
from six import BytesIO


def lambda_handler(event, context):
    client = boto3.client('s3')
    runtime = boto3.client('runtime.sagemaker')

    data = json.loads(json.dumps(event))
    payload = data['instances']

    obj = client.get_object(Bucket='data-science-wine-reviews', Key='word_vectors_idf.csv')
    wine_df = pd.read_csv(obj['Body'])
    wine_df.set_index(['word'], inplace=True)

    word_vectors = []
    for p in payload:
        word_vector_string = wine_df.at[p, 'word_vec_idf']
        word_vector_string = word_vector_string.replace('[', '').replace(r'\n', '').replace(']', '')
        word_vector = np.fromstring(word_vector_string, dtype=float, sep='  ')
        word_vectors.append(word_vector)

    wine_vector = sum(word_vectors) / len(word_vectors)
    wine_vector_output = json.dumps(wine_vector.tolist())

    response = runtime.invoke_endpoint(EndpointName='sagemaker-scikit-learn-2019-07-04-13-00-07-919',
                                       ContentType='application/json',
                                       Body=wine_vector_output)

    def decode(s, encoding="ascii", errors="ignore"):
        return s.decode(encoding=encoding, errors=errors)

    result = json.loads(decode(response['Body'].read()))

    wine_name_lookup = client.get_object(Bucket='data-science-wine-reviews',
                                         Key='nearest_neighbors/data/wine_reviews_select_cols.csv')
    wine_name_lookup = pd.read_csv(wine_name_lookup['Body'])

    recommendation_indices = list(result[1])
    recommendation_indices = [int(n) for n in recommendation_indices]

    recommendations = []
    for i in recommendation_indices:
        suggested_wine = wine_name_lookup.at[i, 'Name']
        descriptors = wine_name_lookup.at[i, 'descriptors']
        recommendations.append([suggested_wine, descriptors])

    return recommendations
