import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_images(images, labels=None, n_images=5, indices=None):
    """Visualizes a random set of images from a training set with their labels.

    Args:
        images (np.ndarray): Array of images (n_samples, x, y).
        labels (np.ndarray): Array of corresponding labels.
        n_images (int, optional): Number of images to visualize. Defaults to 5.
        indices (list, optional): Indices of the images to be visualized
    """

    n_samples = len(images)
    if not indices:
        random_indices = np.random.choice(n_samples, size=min(n_images, n_samples), replace=False)
    else:
        random_indices = indices

    # Just some checks
    assert len(random_indices) == len(set(random_indices)), "Random indices must be unique"
    assert len(random_indices) == n_images, "Number of random indices must match n_images"


    plt.figure(figsize=(10, 5))
    for i, index in enumerate(random_indices):
        plt.subplot(1, min(n_images, n_samples), i + 1)
        plt.imshow(images[index], cmap='gray')
        if labels is not None:
            plt.title(f"Label: {labels[index]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_histograms(image1, image2, lim_x=False):
    """
    Visualizes the histograms of two images side-by-side using seaborn.

    Args:
        image1: The first image as a NumPy array.
        image2: The second image as a NumPy array.
    Returns:
        None
    """
    threshold = [-1300,3000]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Flatten the images for histogram calculation
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Plot the histograms
    sns.histplot(image1_flat, kde=False, ax=axes[0])
    axes[0].set_title('Before scaling')
    axes[0].set_xlabel('Pixel Intensity')
    axes[0].set_ylabel('Frequency')
    if lim_x:
      axes[0].set_xlim(threshold[0],threshold[1])

    sns.histplot(image2_flat, kde=False, ax=axes[1])
    axes[1].set_title('After scaling')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')


    plt.tight_layout()
    plt.show()





