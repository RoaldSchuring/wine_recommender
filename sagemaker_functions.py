train_data = 's3://{}/wine-corpus.txt'.format(bucket)
s3_output_location = 's3://{}/output'.format(bucket)

region_name = boto3.Session().region_name
container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
print('Using SageMaker BlazingText container: {} ({})'.format(container, region_name))


sess = sagemaker.Session()

bt_model = sagemaker.estimator.Estimator(container,
                                         role,
                                         train_instance_count=2,
                                         train_instance_type='ml.c4.2xlarge',
                                         train_volume_size = 5,
                                         train_max_run = 360000,
                                         input_mode= 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)

bt_model.set_hyperparameters(mode="batch_skipgram",
                             epochs=15,
                             min_count=5,
                             sampling_threshold=0.0001,
                             learning_rate=0.05,
                             window_size=5,
                             vector_dim=300,
                             negative_samples=5,
                             batch_size=11, #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
                             evaluation=True,# Perform similarity evaluation on WS-353 dataset at the end of training
                             subwords=False) # Subword embedding learning is not supported by batch_skipgram

train_data = sagemaker.session.s3_input(train_data, distribution='FullyReplicated',
                        content_type='text/plain', s3_data_type='S3Prefix')
data_channels = {'train': train_data}

bt_model.fit(inputs=data_channels, logs=True)

s3 = boto3.resource('s3')
key = bt_model.model_data[bt_model.model_data.find("/", 5)+1:]
s3.Bucket(bucket).download_file(key, 'model.tar.gz')

!tar -xvzf model.tar.gz


from sklearn.preprocessing import normalize
num_points = len(open('vectors.txt','r').read().split('\n'))

first_line = True
index_to_word = []
with open("vectors.txt","r") as f:
    for line_num, line in enumerate(f):
        if first_line:
            dim = int(line.strip().split()[1])
            word_vecs = np.zeros((num_points, dim), dtype=float)
            first_line = False
            continue
        line = line.strip()
        word = line.split()[0]
        vec = word_vecs[line_num-1]
        for index, vec_val in enumerate(line.split()[1:]):
            vec[index] = float(vec_val)
        index_to_word.append(word)
        if line_num >= num_points:
            break
word_vecs = normalize(word_vecs, copy=False, return_norm=False)

names_vecs = list(zip(index_to_word, word_vecs))

names_vecs_filtered = [n for n in names_vecs if n[0] in list(descriptor_mapping['level_3'])]

names_vecs_df = pd.DataFrame(names_vecs_filtered, columns=['word', 'vector'])
names_vecs_df.sort_values(by=['word'], inplace=True)
names_vecs_df.to_csv('word_vectors.csv')
boto3.Session().resource('s3').Bucket(bucket).Object('word_vectors.csv').upload_file('word_vectors.csv')