from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

import cv2
import pandas as pd
import numpy as np


def load(img_path, resize=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if resize is None:
        return img
    
    H, W = img.shape
    target_H, target_W = resize

    if W < target_W or H< target_H:
        print(f"Error: Cannot resize the image because the image is too small")
        return None
    
    x_start = (W - target_W) // 2
    y_start = (H- target_H) // 2

    x_end = x_start + target_W
    y_end = y_start + target_H

    return img[y_start:y_end, x_start:x_end]


def extract_features(img_path, resize=None):
    results = {}
    try:
        img = load(img_path, resize=resize)
        gray_8bit = (img / np.max(img) * 255).astype(np.uint8)

        results['mean'] = img.mean()
        results['std'] = img.std()
        
        results['skewness'] = skew(img.flatten())
        results['kurtosis'] = kurtosis(img.flatten())
        
        results['entropy'] = shannon_entropy(img)
        
        glcm = graycomatrix(gray_8bit, distances=[1], angles=[0, 45, 90, 135], levels=256, 
                            symmetric=True, normed=True)
        # sur tout pixels adjacetns et diagonaux - > (1, 8)

        
        results['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        results['energy'] = np.mean(graycoprops(glcm, 'energy'))
        results['asm'] = np.mean(graycoprops(glcm, 'ASM'))
        results['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
        results['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
        results['correlation'] = np.mean(graycoprops(glcm, 'correlation'))

    except Exception as e:
        for key in ['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'contrast', 'energy', 
                    'asm', 'homogeneity', 'dissimilarity', 'correlation']:
            results[key] = np.nan
        print(f"Error {img_path}: {e}")

    return pd.Series(results)


