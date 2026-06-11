from collections import defaultdict

import tensorflow as tf
import os
import re
from natsort import natsorted

from config import BATCH_SIZE, GRAY


def load_image_mask(image_path, mask_path, img_size):
    """
    Function that takes paths of an image and a mask, reads them, and converts them to tensors

    :param image_path: file path of image
    :param mask_path: file path of mask
    :param img_size: integer size that the image and mask will be resized to in the form img_size x img_size
    :return: 3-dimensional tensor with 3 channels for image, and 3-dimensional tensor with 1 channel for mask
    """
    image = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    mask = tf.io.decode_image(mask, channels=1, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    if GRAY:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    mask = tf.image.resize(mask, [img_size, img_size], method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask


def load_image(image_path, img_size):
    """
    Function that takes paths of an images, reads them, and converts them to tensors

    :param image_path: file path of image
    :param img_size: integer size that the image will be resized to in the form img_size x img_size
    :return: 3-dimensional tensor with 3 channels
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animation=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    if GRAY:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    return image


def get_dataset(folder_path, img_size):
    """
    Takes images and image masks from folder specified, groups them based on which video they are from,
    puts the individual groups into a tf dataset, then returns a list of them

    :param folder_path: Folder path of dataset
    :param img_size: integer size that the images will be resized to in the form img_size x img_size
    :return: List of tf.data.Dataset objects
    """
    original_path = os.path.join(folder_path, "original")
    image_paths_list = defaultdict(list)
    for fname in os.listdir(original_path):
        match = re.search(r"\d+", fname)
        if match:
            index = int(match.group())
            image_paths_list[index].append(os.path.join(original_path, fname))
        for k in image_paths_list:
            image_paths_list[k] = natsorted(image_paths_list[k])

    mask_paths_list = []
    dataset_list = []
    for id in sorted(image_paths_list.keys()):
        image_paths = image_paths_list[id]
        mask_paths = []
        for fname in image_paths:
            name, ext = os.path.splitext(fname)
            name = name.replace("original", "mask")
            mask_paths.append(f"{name}_0{ext}")
        mask_paths_list.append(mask_paths)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(
            lambda image_path, mask_path: load_image_mask(image_path, mask_path, img_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset_list.append(dataset)

    return dataset_list


def get_dataset_no_mask(folder_path, img_size):
    """
    Same as get_dataset but for datasets without masks

    :param folder_path: Folder path of dataset
    :param img_size: integer size that the images will be resized to in the form img_size x img_size
    :return: List of tf.data.Dataset objects
    """
    original_path = os.path.join(folder_path, "original")
    image_paths_list = defaultdict(list)
    for fname in os.listdir(original_path):
        match = re.search(r"\d+", fname)
        if match:
            index = int(match.group())
            image_paths_list[index].append(os.path.join(original_path, fname))
        for k in image_paths_list:
            image_paths_list[k] = natsorted(image_paths_list[k])

    dataset_list = []
    for id in sorted(image_paths_list.keys()):
        image_paths = image_paths_list[id]
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(
            lambda image_path: load_image_mask(image_path, img_size),
            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset_list.append(dataset)

    return dataset_list


def get_tvt_split(folder_path, img_size, augment: bool = True, no_mask: bool = False):
    """
    Takes images (and image masks) from folder specified and turns it into training and validation datasets

    :param folder_path: Folder path of dataset
    :param img_size: integer size that the images will be resized to in the form img_size x img_size
    :param augment: If train dataset should be augmented
    :param no_mask: If datasets have masks
    :return: training and validations dataset: [tf.data.Dataset, tf.data.Dataset]
    """
    if no_mask:
        my_dataset = get_dataset_no_mask(folder_path, img_size)
    else:
        my_dataset = get_dataset(folder_path, img_size)

    def combine_datasets(datasets):
        combined = datasets[0]
        for ds in datasets[1:]:
            combined = combined.concatenate(ds)
        return combined

    n = len(my_dataset)
    train_split = int(0.85 * n)
    train_dataset = combine_datasets(my_dataset[:train_split])
    validate_dataset = combine_datasets(my_dataset[train_split:])
    train_dataset = train_dataset.batch(BATCH_SIZE)
    if augment:
        train_dataset = augment_dataset(train_dataset)
    validate_dataset = validate_dataset.batch(BATCH_SIZE)
    return train_dataset, validate_dataset


def augment_dataset(dataset):
    """
    Takes a dataset and augments it
    :param dataset: tf.data.Dataset
    :return: Augmented dataset
    """
    def augment(image, mask):
        combined = tf.concat([image, mask], axis=-1)
        #combined = random_translate(combined, training=True)

        image = combined[..., :3]
        mask = combined[..., 3:]

        if not GRAY:
            image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
            image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, mask

    random_translate = tf.keras.layers.RandomTranslation(0.4, 0.4, fill_mode="reflect")
    dataset_augmented = dataset.map(augment)

    return dataset_augmented
