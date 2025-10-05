import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import logging
logger = logging.getLogger(__name__)

from imagerec.common import albumentation_transforms


def denormalize(image_tensor):
    """Assumes ImageNet mean/std normalization"""

    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(image, 0, 1)

def visualize_scroll(
    image_path,
    image_size,
    config_path,
    num_samples=5
):
    """
    Visualize image augmentations interactively using scroll.

    Args:
        image_path (str or Path): Path to the input image file.
        config_path (str or Path): Path to augmentation configuration file.
        image_size (int): Target size for augmentation transformations.
        num_samples (int, optional): Number of augmented samples to generate (default: 5).

    Notes:
        - Use the mouse scroll wheel to switch between augmentations.
        - Displays one augmented image at a time.
    """
    logger.info(f"Loading image for scroll visualization: {image_path}")
    transform = albumentation_transforms.TrainTransform(
        image_size=image_size,
        config_path=config_path
    )

    image = PIL.Image.open(image_path)

    if isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = np.array(image)

    logger.info(f"Generating {num_samples} augmented samples.")

    # generate augmented images
    images = []
    for _ in range(num_samples):
        augmented = transform(image=image_np)
        img = denormalize(augmented)
        images.append(img)

    index = [0]

    fig, ax = plt.subplots()
    img_display = ax.imshow(images[index[0]])
    ax.axis("off")
    title = ax.set_title(f"Augmentation {index[0]+1}/{num_samples}")

    def on_scroll(event):
        if event.button == 'down':
            index[0] = (index[0] + 1) % num_samples
        elif event.button == 'up':
            index[0] = (index[0] - 1) % num_samples

        img_display.set_data(images[index[0]])
        title.set_text(f"Augmentation {index[0]+1}/{num_samples}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

def visualize_grid(
        image_path,
        image_size,
        config_path,
        num_samples=5
):
    """
    Visualize multiple augmented images in a paginated grid.

    Args:
        image_path (str or Path): Path to the input image file.
        config_path (str or Path): Path to augmentation configuration file.
        image_size (int): Target size for augmentation transformations.
        num_samples (int, optional): Number of augmented samples to generate (default: 5).

    Notes:
        - Augmented images are shown in a grid (default 4x4, 16 images per page).
        - Use mouse scroll wheel to switch between pages if samples > 16.
    """
    cols = 4
    rows = 4
    max_per_page = rows * cols
    pages = (num_samples + max_per_page - 1) // max_per_page
    current_page = [0]

    transform = albumentation_transforms.TrainTransform(
        image_size=image_size,
        config_path=config_path
    )

    image = PIL.Image.open(image_path)

    if isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = np.array(image)

    # generate all augmented images
    augmented_images = []
    for _ in range(num_samples):
        augmented = transform(image=image_np)
        augmented_images.append(denormalize(augmented))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"Page 1 / {pages}", fontsize=16)

    axes = axes.flatten()

    def show_page(page):
        start = page * max_per_page
        fig.suptitle(f"Page {page + 1} / {pages}", fontsize=16)

        for ax in axes:
            ax.clear()
            ax.axis("off")

        for i, ax in enumerate(axes):
            idx = start + i
            if idx < len(augmented_images):
                ax.imshow(augmented_images[idx])
                ax.set_title(f"Sample {idx + 1}")
                ax.axis("off")

        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.button == 'up':
            current_page[0] = (current_page[0] - 1 + pages) % pages
        elif event.button == 'down':
            current_page[0] = (current_page[0] + 1) % pages
        show_page(current_page[0])

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    show_page(current_page[0])
    plt.tight_layout()
    plt.show()