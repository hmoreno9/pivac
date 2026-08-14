import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from cryptography.fernet import Fernet
import easyocr

# Importación de los nuevos submódulos especializados
from megadescriptor_model import MegaDescriptorWrapper
from traditional_features import TraditionalMuzzleExtractor
from spoof_model import SpoofDetectorWrapper


class CattleBiometricEngine:
    def __init__(self, encryption_key: bytes = None):

        # 1. Inicializar OCR para lectura de caravana
        self.ocr_reader = easyocr.Reader(['en', 'es'], gpu=torch.cuda.is_available())

        # 2. Inicializar modelo para Embeddings (ResNet18 Base)
        weights = ResNet18_Weights.DEFAULT
        self.feature_extractor = resnet18(weights=weights)
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Submódulos Especializados (Deep, Textura LBP/HOG y Anti-Spoofing)
        self.mega_descriptor = MegaDescriptorWrapper()
        self.texture_extractor = TraditionalMuzzleExtractor()
        self.spoof_detector = SpoofDetectorWrapper()

        # 4. Configuración de Cifrado AES (Fernet)
        if encryption_key is None:
            self.cipher_key = Fernet.generate_key()
        else:
            self.cipher_key = encryption_key
        self.cipher = Fernet(self.cipher_key)

    # ==========================================
    # 0. RECORTE DE ZONA NASAL (MUZZLE ROI CROP)
    # ==========================================
    def crop_muzzle_roi(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Garantiza que la extracción se realice SOLO sobre la piel del morro
        y elimine la cabeza, ojos y pelaje de la vaca.
        """
        h, w, _ = img_bgr.shape
        # Recorte estratégico enfocado en la región nasolabial
        y_start, y_end = int(h * 0.35), int(h * 0.85)
        x_start, x_end = int(w * 0.20), int(w * 0.80)
        
        muzzle_roi = img_bgr[y_start:y_end, x_start:x_end]
        return muzzle_roi if muzzle_roi.size > 0 else img_bgr

    # ==========================================
    # 1. PREPROCESAMIENTO Y FILTRADO DE MORRO
    # ==========================================
    def preprocess_muzzle(self, img_bytes: bytes) -> np.ndarray:
        """
        Decodifica la imagen, realiza el crop estricto del morro,
        realza el contraste local de las crestas/surcos (CLAHE)
        y aplica filtros de Gabor para destacar minucias.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        # 1. Recorte estricto al área del morro
        muzzle_crop = self.crop_muzzle_roi(img)

        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(muzzle_crop, cv2.COLOR_BGR2GRAY)

        # 3. CLAHE para destacar crestas y poros
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Filtro de Gabor para alineación de surcos/crestas
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)

        return filtered

    # ==========================================
    # 2. EXTRACCIÓN HÍBRIDA DE VECTOR BIOMÉTRICO
    # ==========================================
    def extract_vector(self, processed_img: np.ndarray) -> np.ndarray:
        """
        Convierte el morro preprocesado en un vector denso combinado:
        Embedding Profundo (ResNet18/Siamesa) + Textura Micro (LBP + HOG).
        """
        # Vector Deep (ResNet18)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        tensor_img = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            deep_embedding = self.feature_extractor(tensor_img).numpy().squeeze()
        
        deep_norm = deep_embedding / (np.linalg.norm(deep_embedding) + 1e-10)

        # Vector de Textura Tradicional (LBP + HOG)
        texture_vector = self.texture_extractor.extract_combined_vector(img_rgb)

        # Combinación de características
        hybrid_vector = np.hstack([deep_norm, texture_vector])
        
        # Normalización L2 final
        final_norm = hybrid_vector / (np.linalg.norm(hybrid_vector) + 1e-10)
        return final_norm.astype(np.float32)

    # ==========================================
    # 2b. VALIDACIÓN ANTI-SPOOFING (PRUEBA DE VIDA)
    # ==========================================
    def check_spoofing(self, img_bytes: bytes) -> dict:
        """
        Evalúa si la imagen enviada proviene de un animal vivo en campo.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"is_live_animal": False, "confidence_real": 0.0}
            
        muzzle_crop = self.crop_muzzle_roi(img)
        return self.spoof_detector.check_spoofing(muzzle_crop)

    # ==========================================
    # 3. ENCRIPTACIÓN DEL MORRO Y PATRÓN
    # ==========================================
    def encrypt_data(self, data_bytes: bytes) -> str:
        """
        Encripta datos sensibles (vector o imagen) con AES-256.
        """
        encrypted = self.cipher.encrypt(data_bytes)
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_data(self, encrypted_str: str) -> bytes:
        """
        Descifra datos encriptados.
        """
        raw_encrypted = base64.b64decode(encrypted_str.encode('utf-8'))
        return self.cipher.decrypt(raw_encrypted)

    # ==========================================
    # 4. OCR DE CARAVANA
    # ==========================================
    def process_tag_ocr(self, tag_img_bytes: bytes) -> str:
        """
        Extrae el texto/número de la imagen de la caravana.
        """
        nparr = np.frombuffer(tag_img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        
        results = self.ocr_reader.readtext(img, detail=0)
        extracted_text = "".join(results).replace(" ", "").upper()
        return extracted_text
