import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

#Combinar el vector de la red con descriptores de textura clásica (Local Binary Patterns y HOG) 
# garantiza que la comparación dependa de las huellas del morro y no del color de la piel.

class TraditionalMuzzleExtractor:
    def __init__(self, lbp_radius: int = 3, lbp_points: int = 24):
        self.radius = lbp_radius
        self.points = lbp_points

    def preprocess_gray_muzzle(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento óptimo para resaltar crestas y surcos:
        Grayscale + CLAHE (Equalización adaptativa de contraste)
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Reducción de ruido preservando bordes
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Aumento de contraste adaptativo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        return enhanced

    def extract_lbp_histogram(self, gray_muzzle: np.ndarray) -> np.ndarray:
        """
        Extrae Local Binary Patterns (LBP) para describir la textura micro de los surcos.
        """
        lbp = local_binary_pattern(gray_muzzle, self.points, self.radius, method="uniform")
        
        # Histograma normalizado de patrones LBP
        n_bins = self.points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)

    def extract_hog_features(self, gray_muzzle: np.ndarray) -> np.ndarray:
        """
        Extrae Histogram of Oriented Gradients (HOG) para capturar la orientación de las crestas.
        """
        resized = cv2.resize(gray_muzzle, (128, 128))
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )
        return features.astype(np.float32)

    def extract_combined_vector(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Devuelve la combinación de LBP y HOG normalizados.
        """
        enhanced = self.preprocess_gray_muzzle(image_bgr)
        lbp_hist = self.extract_lbp_histogram(enhanced)
        hog_feat = self.extract_hog_features(enhanced)
        
        combined = np.hstack([lbp_hist, hog_feat])
        norm = np.linalg.norm(combined)
        return (combined / (norm + 1e-10)).astype(np.float32)
