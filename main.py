import cv2
import os
import numpy as np

def setdata(foldername):
    images = []
    labels = []
    image_paths = []

    for folder in os.listdir(foldername):
        folder_path = os.path.join(foldername, folder)
        if not os.path.isdir(folder_path):
            continue

        for imgfile in os.listdir(folder_path):
            img_path = os.path.join(folder_path, imgfile)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (128, 128)) / 255.0
            img = img.astype(np.float32)
            img_vector = img.reshape(-1, 1)  # 2D column vector
            images.append(img_vector)
            labels.append(folder)
            image_paths.append(img_path)

    if len(images) == 0:
        raise ValueError("No valid images found in dataset!")

    return images, labels, image_paths

def average(images):
    mean = np.mean(images, axis=0)
    diff_images = [img - mean for img in images]
    return diff_images, mean

def bigA(diff_images):
    return np.hstack(diff_images)  # Matrix A: each column is a diff image

def eigenface(images, num_components=None):
    diff_images, mean = average(images)
    A = bigA(diff_images)

    # Covariance trick
    L = A.T @ A  # small matrix
    eigvals, eigvecs_small = np.linalg.eig(L)
    eigvals = eigvals.real
    eigvecs_small = eigvecs_small.real

    # Sort eigenvalues descending
    idx = np.argsort(-eigvals)
    eigvecs_small = eigvecs_small[:, idx]

    if num_components is not None:
        eigvecs_small = eigvecs_small[:, :num_components]

    # Project back to high-dim space
    eigfaces = A @ eigvecs_small
    eigfaces = eigfaces / np.linalg.norm(eigfaces, axis=0)
    return eigfaces, mean

def compute_weights(images, mean, eigfaces):
    weights = []
    for img in images:
        w = eigfaces.T @ (img - mean)  # Result: (num_components, 1)
        weights.append(w)
    return weights

def eucdistance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def mainprog(img_input_path, dataset_path, num_components=50):
    if not os.path.exists(img_input_path):
        raise FileNotFoundError(f"Input image not found: {img_input_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Step 1: Load dataset
    images, labels, image_paths = setdata(dataset_path)

    # Step 2: Compute eigenfaces
    eigfaces, mean = eigenface(images, num_components)

    # Step 3: Compute weights for dataset
    weights_dataset = compute_weights(images, mean, eigfaces)

    # Step 4: Load input image
    img = cv2.imread(img_input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Input image could not be read!")

    img = cv2.resize(img, (128, 128)) / 255.0
    img = img.astype(np.float32)
    img_vector = img.reshape(-1, 1)

    # Step 5: Compute weight for input image
    input_weight = eigfaces.T @ (img_vector - mean)

    # Step 6: Compare with all dataset weights
    distances = [eucdistance(input_weight, w) for w in weights_dataset]

    min_idx = int(np.argmin(distances))  # Ensure it's scalar
    closest_img_path = image_paths[min_idx]

    # Step 7: Save closest match
    output_path = os.path.abspath("closest_match.jpg")
    matched_img = cv2.imread(closest_img_path)
    if matched_img is None:
        raise ValueError("Closest match image could not be read!")

    cv2.imwrite(output_path, matched_img)
    print(f"Closest match found: {closest_img_path}")
    return output_path

# Optional: Example usage
if __name__ == "__main__":
    dataset_path = "path_to_your_dataset"         # Ganti ke folder dataset
    img_input_path = "path_to_input_image.jpg"    # Ganti ke file input

    try:
        result_path = mainprog(img_input_path, dataset_path)
        print(f"Hasil disimpan di: {result_path}")
    except Exception as e:
        print(f"Error: {e}")
