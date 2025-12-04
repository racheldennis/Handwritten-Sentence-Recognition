import matplotlib.pyplot as plt

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

def show_transformations(raw_image, transformed_image):
    plt.figure(figsize=(20, 4))

    # raw image
    plt.subplot(1, 2, 1)
    plt.title("Before Transforms")
    plt.imshow(raw_image, cmap="gray")
    plt.axis("off")

    # transformed image
    plt.subplot(1, 2, 2)
    plt.title("After Transforms")
    image = unnormalize(transformed_image).permute(1, 2, 0).squeeze() # (C, H, W) -> (H, W, C) -> (H, W)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.show(block=False)

def show_predictions(images, predictions, labels, n=5):
    for i in range(n):
        image = unnormalize(images[i]).squeeze() # (1, H, W) -> (H, W)

        plt.figure(figsize=(12, 2))
        plt.imshow(image.cpu(), cmap="gray")
        plt.title(f"Predicted: {predictions[i]}\nTrue: {labels[i]}")
        plt.axis("off")  
        plt.show(block=False)
