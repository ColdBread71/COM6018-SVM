import sys

import joblib

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def create_dataset(images, pca_model):
    """Create a dataset and preprocess it."""
    # Expand the image
    x = images.reshape(images.shape[0], -1)

    """
    If the test data itself is standardized and the training set is not used, 
    the data distribution may be slightly different
    """
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    x = pca_model.transform(x)

    return x


def parse_args():
    """Parse command line parameters"""
    args = sys.argv[1:]

    model_file_name = args[0]
    image_size = int(args[1])

    return model_file_name, image_size


if __name__ == '__main__':
    model_file_name, image_size = parse_args()

    saved_dict = joblib.load(model_file_name)
    model = saved_dict.get('model')
    pca_model = saved_dict.get('pca')

    data_dict = joblib.load('./datasets/eval1.joblib')
    x_test = create_dataset(data_dict[image_size]['x_test'], pca_model)
    y_test = data_dict[image_size]['y_test']

    # test
    y_pred = model.predict(x_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100}%')
