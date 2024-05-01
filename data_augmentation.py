#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import argparse
import lmdb
import random
from PIL import Image
from tqdm import tqdm


@dataclass
class AugmentedImage:
    image: bytes
    label: str
    name: str


def augment_ds(source_path: str, target_path: str, labels_path: str, output_labels: str,
               augmentation_function=lambda key, image, label, random_image, random_label: augment_images(key, image, label, random_image, random_label), n: int = 0):
    """
       Augmentates dataset.
       :param source_path (str): path leading to source lmdb file
       :param target_path (str): path leading to output lmdb file
       :param augmentation_function (callable): function to augment images
       :param n (int): number of images to augment
       """
    with lmdb.open(source_path) as source_db, lmdb.open(target_path,
                                                        map_size=source_db.info()['map_size']) as target_db:
        with source_db.begin() as source_tx, target_db.begin(write=True) as target_tx:
            label_file = open(labels_path, 'r')
            output_labels = open(output_labels, 'w')

            label_dict = {}

            # Store labels in a dictionary for easy random selection
            for line in label_file:
                key, label = line.strip().split(' 0 ')
                label_dict[key] = label

            i = 0
            for key, label in label_dict.items():
                print(f"\raugmenting {i} {key}",end="")
                i = i+1
                image = source_tx.get(key.encode())

                # Choose a random image and its label
                random_key, random_label = random.choice(list(label_dict.items()))
                random_image = source_tx.get(random_key.encode())

                augmented_images = augmentation_function(key, image, label, random_image, random_label)

                for img in augmented_images:
                    target_tx.put(
                        key=img.name.encode(),
                        value=img.image
                    )
                    output_labels.write(f'{img.name} 0 {img.label}\n')

                output_labels.write(f'{key} 0 {label}\n')
                target_tx.put(key=key.encode(), value=image)
            print("\n")

            label_file.close()
            output_labels.close()





def augment_images(key: str, image_bytes: bytes, label: str, other_image_bytes: bytes, other_label: str):
    augmented_images = []

    # Decode image bytes into numpy arrays
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    other_image = np.frombuffer(other_image_bytes, dtype=np.uint8)
    other_image = cv2.imdecode(other_image, cv2.IMREAD_COLOR)

    # Resize other_image to match the dimensions of image
    other_image = cv2.resize(other_image, (image.shape[1], image.shape[0]))

    # Define the skew parameters
    rotate_x = np.random.uniform(-0.05, 0.05)  # Skew factor in the x-direction
    rotate_y = np.random.uniform(-0.05, 0.05)  # Skew factor in the y-direction

    # Define the transformation matrix
    M = np.float32([[1, rotate_x, 0], [rotate_y, 1, 0]])
    augmented_image_rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Apply Gaussian noise augmentation
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    augmented_image_noise = cv2.add(image, noise)

    # Apply color mask augmentation with a single random color
    mask_color = np.random.randint(0, 256, 3, dtype=np.uint8)  # Generate a random color
    mask = np.ones_like(
        image) * mask_color  # Create a mask with the same size as the original image filled with the random color
    augmented_image_mask = cv2.addWeighted(image.astype(np.float32), 0.7, mask.astype(np.float32), 0.3, 0).astype(np.uint8)  # Apply the mask to the original image

    # Define the skew parameters
    skew_x = np.random.uniform(0.3,1)  # Skew factor in the x-direction


    # Define the transformation matrix
    M = np.float32([[1, skew_x, 0], [0, 1, 0]])

    # Apply the skew transformation to the image
    augmented_image_skewed = cv2.warpAffine(image, M, (int(image.shape[1] + abs(skew_x * image.shape[0])), image.shape[0]))

    # Apply resizing augmentation (stretch or shrink)
    scale_factor = np.random.uniform(0.5, 1.5)  # Random scale factor between 0.5 and 1.5
    new_width = int(image.shape[1] * scale_factor)
    new_height = image.shape[0]  # Maintain original height
    augmented_image_resize = image
    if new_width >= 1:
        augmented_image_resize = cv2.resize(image, (new_width, new_height))
    else:
        print(f"\nresizing {key}: original w={image.shape[1]} new w={new_width}\n")

    # Apply Gaussian square augmentation
    size = np.random.randint(1, min(41, min(image.shape[0], image.shape[1])))
    top = np.random.randint(0, image.shape[0] - size)
    left = np.random.randint(0, image.shape[1] - size)
    noise = np.random.normal(0, 25, (size, size, 3)).astype(np.uint8)
    augmented_image_gaussian_square = image.copy()
    augmented_image_gaussian_square[top:top+size, left:left+size] = noise

    # Mixup augmentation
    lambda_val = np.random.beta(1, 1)  # Random lambda value from beta distribution
    mixup_label = label if lambda_val > 0.5 else other_label
    mixed_image = cv2.addWeighted(image, lambda_val, other_image, 1 - lambda_val, 0)

    # CutMix augmentation - Divide images with a horizontal line
    cut_ratio = np.random.rand()  # Random cut ratio
    cut_width = int(cut_ratio * image.shape[1])  # Width for cut

    # Cut images and labels
    cut_image_1 = image[:, :cut_width]
    cut_image_2 = other_image[:, cut_width:]
    cut_label_1 = label.split()[:cut_width]
    cut_label_2 = other_label.split()[cut_width:]

    # Mix cut images and labels
    cutmix_image = np.concatenate((cut_image_1, cut_image_2), axis=1)
    cutmix_label = ' '.join(cut_label_1 + cut_label_2)


    # Convert augmented images back to bytes
    _, img_encoded_rotate = cv2.imencode('.jpg', augmented_image_rotate)
    _, img_encoded_noise = cv2.imencode('.jpg', augmented_image_noise)
    _, img_encoded_mask = cv2.imencode('.jpg', augmented_image_mask)
    _, img_encoded_skewed = cv2.imencode('.jpg', augmented_image_skewed)
    _, img_encoded_resize = cv2.imencode('.jpg', augmented_image_resize)
    _, img_encoded_gaussian_square = cv2.imencode('.jpg', augmented_image_gaussian_square)
    _, img_encoded_mixed = cv2.imencode('.jpg', mixed_image)
    _, img_encoded_cutmix = cv2.imencode('.jpg', cutmix_image)

    names = [f"{key.split('.')[0]}_rotate.jpg", f"{key.split('.')[0]}_noise.jpg", f"{key.split('.')[0]}_mask.jpg",
             f"{key.split('.')[0]}_skewed.jpg", f"{key.split('.')[0]}_resize.jpg",
             f"{key.split('.')[0]}_gaussian_square.jpg",
             f"{key.split('.')[0]}_mixup.jpg", f"{key.split('.')[0]}_cutmix.jpg",]
    images = [img_encoded_rotate.tobytes(), img_encoded_noise.tobytes(), img_encoded_mask.tobytes(),
              img_encoded_skewed.tobytes(), img_encoded_resize.tobytes(), img_encoded_gaussian_square.tobytes(),
              img_encoded_mixed.tobytes(), img_encoded_cutmix.tobytes()]
    labels = [label, label, label, label, label, label, mixup_label, cutmix_label]

    for name, image, label in zip(names, images, labels):
        augmented_images.append(AugmentedImage(image=image, label=label, name=name))

    return augmented_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment images in the LMDB database.")
    parser.add_argument("--database-path", type=str, help="Path to the LMDB database.")
    parser.add_argument("--label-file", type=str, help="Path to the label file.")
    parser.add_argument("--output-db-path", type=str, help="Path to the output LMDB database.")
    parser.add_argument("--output-label-file", type=str, help="Path to the output label file.")
    args = parser.parse_args()

    augment_ds(args.database_path, args.output_db_path, args.label_file, args.output_label_file)

