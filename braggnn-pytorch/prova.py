import keras
import tensorflow as tf


def load() -> tuple[tf.data.Dataset, tf.data.Dataset]:

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


    test_images = tf.data.Dataset.from_tensor_slices(test_images)
    test_labels = tf.data.Dataset.from_tensor_slices(test_labels)
    train_images = tf.data.Dataset.from_tensor_slices(train_images)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

    def preprocess_img(img : tf.Tensor) -> tf.Tensor:
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)

        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = (img - mean) / std
        return img
    
    train_images = train_images.map(preprocess_img)
    test_images = test_images.map(preprocess_img)

    # trasform into tensor
    train_dataser = tf.data.Dataset.zip((train_images, train_labels))
    validation_dataset = tf.data.Dataset.zip((test_images, test_labels))

    return train_dataser, validation_dataset

_, dataset = load()
