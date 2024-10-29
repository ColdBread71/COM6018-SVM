import sys

import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.random.seed(2023)


def generate_crop_images(images, target_size=90):
    """Generate cropped images"""
    crop_images = []

    for image in images:
        # Note that the length and width are different before cutting,
        # so x and y need to be calculated separately.
        x = np.random.randint(0, image.shape[0] - target_size + 1)
        y = np.random.randint(0, image.shape[1] - target_size + 1)

        # Crop to target size
        crop_image = image[x: x + target_size, y: y + target_size]
        crop_images.append(crop_image)

    return crop_images


def generate_rotated_images(images, add_noise=False):
    """Generate a rotated image"""
    rotated_images = []
    rotated_labels = []

    for image in images:
        # Randomly generate a direction
        #  0—>upright, 1->left, 2->inverted, 3->right
        label = np.random.randint(0, 4)
        rotated_labels.append(label)

        rotated_image = np.rot90(image, k=label)
        # use noise
        if add_noise:
            rotated_image += np.random.normal(scale=0.01, size=rotated_image.shape)

        rotated_images.append(rotated_image)

    return np.asarray(rotated_images), np.asarray(rotated_labels)


def augment_images(images, labels):
    """Flip and augment data"""
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):

        augmented_images.append(image)
        augmented_labels.append(label)

        # 0, 180 pictures, flipped left and right.
        if label in [0, 2]:
            flipped_image = np.fliplr(image)

            augmented_images.append(flipped_image)
            augmented_labels.append(label)
        # Picture of 90, 270, flipped upside down
        else:
            flipped_image = np.flipud(image)

            augmented_images.append(flipped_image)
            augmented_labels.append(label)

    return np.asarray(augmented_images), np.asarray(augmented_labels)


# Hyperparameters.
HYPER_PARAMS = {
    90: {
        'add_noise': True,
        'pca_n_components': 50,
        'svm_c': 3,
        'svm_gamma': 'scale'  # 默认值, 因为下面改动了所以留个参数位置.
    },
    50: {
        'add_noise': True,
        'pca_n_components': 50,
        'svm_c': 5,
        'svm_gamma': 'scale'
    },
    30: {
        'add_noise': False,
        'pca_n_components': 75,
        'svm_c': 7,
        'svm_gamma': 0.09
    }
}


def create_dataset(images, labels, n_components):

    x = images.reshape(images.shape[0], -1)

    # standardization
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # reduction
    pca_model = PCA(n_components=n_components, random_state=2023)
    x = pca_model.fit_transform(x)

    return x, labels, pca_model


def create_model(c, gamma):
    model = SVC(kernel='rbf',
                C=c,
                gamma=gamma)

    return model


def parse_args():

    args = sys.argv[1:]

    training_data_file_name = args[0]
    model_file_name = args[1]
    image_size = int(args[2])

    return training_data_file_name, model_file_name, image_size


if __name__ == '__main__':
    training_data_file_name, model_file_name, image_size = parse_args()
    hyper_param = HYPER_PARAMS.get(image_size)
    print('Hyper parameters: ', hyper_param)

    images, _ = joblib.load(training_data_file_name)

    images = generate_crop_images(images, image_size)
    images, labels = generate_rotated_images(images, hyper_param.get('add_noise'))
    images, labels = augment_images(images, labels)

    x, y, pca_model = create_dataset(images, labels, hyper_param.get('pca_n_components'))

    # train
    model = create_model(hyper_param.get('svm_c'), hyper_param.get('svm_gamma'))
    model.fit(x, y)
    print(f'Accuracy: {accuracy_score(y, model.predict(x)) * 100}%')

    # save model
    saved_dict = {
        'model': model,
        'pca': pca_model
    }
    joblib.dump(saved_dict, model_file_name)
