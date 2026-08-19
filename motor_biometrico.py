import base64
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from cryptography.fernet import Fernet


class CattleBiometricEngine:
    def __init__(self, encryption_key: bytes = None):
        # Desactivar gradientes globalmente para reducir consumo de RAM
        torch.set_grad_enabled(False)

        # Referencias nulas para carga bajo demanda (Lazy Loading)
        self.ocr_reader = None
        self.feature_extractor = None
        self.mega_descriptor = None
        self.texture_extractor = None
        self.spoof_detector = None

        # Transformación para PyTorch
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Configuración de Cifrado AES (Fernet)
        if encryption_key is None:
            self.cipher_key = Fernet.generate_key()
        else:
            self.cipher_key = encryption_key
        self.cipher = Fernet(self.cipher_key)

    # ==========================================
    # PROPIDATIVE / LAZY LOADERS (Ahorro de RAM)
    # ==========================================
    def _get_ocr(self):
        if self.ocr_reader is None:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        return self.ocr_reader

    def _get_feature_extractor(self):
        if self.feature_extractor is None:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            self.feature_extractor = model
        return self.feature_extractor

    def _get_mega_descriptor(self):
        if self.mega_descriptor is None:
            from megadescriptor_model import MegaDescriptorWrapper
            self.mega_descriptor = MegaDescriptorWrapper()
        return self.mega_descriptor

    def _get_texture_extractor(self):
        if self.texture_extractor is None:
            from traditional_features import TraditionalMuzzleExtractor
            self.texture_extractor = TraditionalMuzzleExtractor()
        return self.texture_extractor

    def _get_spoof_detector(self):
        if self.spoof_detector is None:
            from spoof_model import SpoofDetectorWrapper
            self.spoof_detector = SpoofDetectorWrapper()
        return self.spoof_detector

    # ==========================================
    # 0. RECORTE DE ZONA NASAL (MUZZLE ROI CROP)  mantener la imagen entera de la vaca
    # ==========================================
    def crop_muzzle_roi_cara(self, img_bgr: np.ndarray):
        h, w, _ = img_bgr.shape
        # Proporciones específicas para enfocar el morro en la toma
        y_start, y_end = int(h * 0.45), int(h * 0.90)
        x_start, x_end = int(w * 0.25), int(w * 0.75)

        muzzle_roi = img_bgr[y_start:y_end, x_start:x_end]
        if muzzle_roi.size == 0:
            return img_bgr, (0, 0, w, h)
        
        return muzzle_roi, (x_start, y_start, x_end - x_start, y_end - y_start)

    def crop_muzzle_roi(self, img_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Garantiza que la extracción se realice SOLO sobre la piel del morro.
        Retorna la imagen recortada y sus coordenadas Bounding Box (x, y, w, h).
        """
        h, w, _ = img_bgr.shape
        y_start, y_end = int(h * 0.35), int(h * 0.85)
        x_start, x_end = int(w * 0.20), int(w * 0.80)

        muzzle_roi = img_bgr[y_start:y_end, x_start:x_end]
        if muzzle_roi.size == 0:
            return img_bgr, (0, 0, w, h)
        return muzzle_roi, (x_start, y_start, x_end - x_start, y_end - y_start)
    
    def crop_muzzle_roi_old(self, img_bgr: np.ndarray) -> np.ndarray:
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
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")

        muzzle_crop, _ = self.crop_muzzle_roi(img)
        gray = cv2.cvtColor(muzzle_crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)

        return filtered
    
    def preprocess_muzzle_old(self, img_bytes: bytes) -> np.ndarray:
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
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)

        return filtered

    # ==========================================
    # 2. EXTRACCIÓN HÍBRIDA DE VECTOR BIOMÉTRICO
    # ==========================================
    def extract_vector(self, processed_img: np.ndarray) -> np.ndarray:
        extractor = self._get_feature_extractor()
        texture_ext = self._get_texture_extractor()

        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        tensor_img = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            deep_embedding = extractor(tensor_img).numpy().squeeze()

        deep_norm = deep_embedding / (np.linalg.norm(deep_embedding) + 1e-10)
        texture_vector = texture_ext.extract_combined_vector(img_rgb)

        hybrid_vector = np.hstack([deep_norm, texture_vector])
        final_norm = hybrid_vector / (np.linalg.norm(hybrid_vector) + 1e-10)
        return final_norm.astype(np.float32)
    
    def extract_vector_old(self, processed_img: np.ndarray) -> np.ndarray:
        """
        Convierte el morro preprocesado en un vector denso combinado:
        Embedding Profundo (ResNet18/Siamesa) + Textura Micro (LBP + HOG).
        """
        extractor = self._get_feature_extractor()
        texture_ext = self._get_texture_extractor()

        # Vector Deep (ResNet18)
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        tensor_img = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            deep_embedding = extractor(tensor_img).numpy().squeeze()

        deep_norm = deep_embedding / (np.linalg.norm(deep_embedding) + 1e-10)

        # Vector de Textura Tradicional (LBP + HOG)
        texture_vector = texture_ext.extract_combined_vector(img_rgb)

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
        spoof_det = self._get_spoof_detector()
        return spoof_det.check_spoofing(muzzle_crop)

    # ==========================================
    # 3. ANOTACIÓN VISUAL DE MINUCIAS Y MORRO (1 FOTO)
    # ==========================================
    def get_annotated_muzzle(self, image_bytes: bytes):
        """Genera una imagen BGR codificada en base64 con el ROI y las minucias (puntos clave) solapadas."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, 0

        # 1. Delimitar el ROI del morro
        h, w, _ = img.shape
        x_min, y_min = int(w * 0.20), int(h * 0.35)
        x_max, y_max = int(w * 0.80), int(h * 0.85)
        crop_w, crop_h = x_max - x_min, y_max - y_min

        # Dibujar recuadro del ROI sobre la imagen original
        cv2.rectangle(
            img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
        )  # Verde

        # 2. Extraer minucias con ORB dentro de la región recortada
        muzzle_roi = img[y_min:y_max, x_min:x_max]
        gray_roi = cv2.cvtColor(muzzle_roi, cv2.COLOR_BGR2GRAY)

        # Aplicar CLAHE para resaltar relieves antes de detectar puntos
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_roi = clahe.apply(gray_roi)

        orb = cv2.ORB_create(nfeatures=600)
        keypoints = orb.detect(enhanced_roi, None)

        # Reajustar coordenadas de keypoints a la imagen completa
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x_min, kp.pt[1] + y_min)

        # Dibujar minucias (puntos en rojo)
        img_annotated = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            color=(0, 0, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Codificar a base64 para envío directo a la app web
        _, buffer = cv2.imencode('.jpg', img_annotated)
        img_b64 = f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'

        bbox_dict = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
        }
        return img_b64, bbox_dict, len(keypoints)


    def get_annotated_muzzle_old(self, image_bytes: bytes) -> tuple[bytes, dict, int]:
        """
        Segmenta el morro, extrae minucias/puntos clave y dibuja
        las marcas solapadas sobre la imagen original.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagen no válida.")

        # 1. Delimitar el área del morro (ROI)
        muzzle_roi, (x, y, w, h) = self.crop_muzzle_roi(img)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) # Recrodrado Verde ROI

        # 2. Extraer puntos clave / minucias con ORB/SIFT en la ROI
        gray = cv2.cvtColor(muzzle_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        orb = cv2.ORB_create(nfeatures=400)
        keypoints = orb.detect(enhanced, None)

        # Reajustar coordenadas de puntos clave a la imagen global
        for kp in keypoints:
            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

        # 3. Solapar minucias (Círculos verdes con núcleo rojo)
        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(img, pt, 4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, pt, 1, (0, 0, 255), -1, cv2.LINE_AA)

        # Codificar a JPEG
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        bbox_dict = {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}

        return buffer.tobytes(), bbox_dict, len(keypoints)

    # ==========================================
    # 4. MATCHING VISUAL 1 A 1 (FOTO CAMPO VS FOTO BD)
    # ==========================================
    def get_matched_visual(
        self, img_query_bytes: bytes, img_db_bytes: bytes, top_matches=40
    ) -> str:
        """Genera una imagen compuesta que muestra el solapamiento 1 a 1 de coincidencias entre la imagen de prueba y la imagen en BD."""
        np_q = np.frombuffer(img_query_bytes, np.uint8)
        np_db = np.frombuffer(img_db_bytes, np.uint8)

        img_q = cv2.imdecode(np_q, cv2.IMREAD_COLOR)
        img_db = cv2.imdecode(np_db, cv2.IMREAD_COLOR)

        if img_q is None or img_db is None:
            return ''

        # Aplicar crop ROI a ambas imágenes
        crop_q = self.crop_muzzle_roi(img_q)
        crop_db = self.crop_muzzle_roi(img_db)

        gray_q = cv2.cvtColor(crop_q, cv2.COLOR_BGR2GRAY)
        gray_db = cv2.cvtColor(crop_db, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=800)
        kp1, des1 = orb.detectAndCompute(gray_q, None)
        kp2, des2 = orb.detectAndCompute(gray_db, None)

        if des1 is None or des2 is None:
            return ''

        # Asociación de minucias con Hamming Distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Generar imagen lado a lado con líneas de conexión
        matched_img = cv2.drawMatches(
            crop_q,
            kp1,
            crop_db,
            kp2,
            matches[:top_matches],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        _, buffer = cv2.imencode('.jpg', matched_img)
        return f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'


    def generate_matching_visual_old(self, img_query_bytes: bytes, img_enrolled_bytes: bytes) -> tuple[bytes, int]:
        """
        Genera la imagen comparativa 1:1 solapando y uniendo los puntos
        de coincidencia entre la foto de campo y la foto de la BD.
        """
        np_q = np.frombuffer(img_query_bytes, np.uint8)
        np_e = np.frombuffer(img_enrolled_bytes, np.uint8)
        
        img1 = cv2.imdecode(np_q, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np_e, cv2.IMREAD_COLOR)

        roi1, _ = self.crop_muzzle_roi(img1)
        roi2, _ = self.crop_muzzle_roi(img2)

        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            # Fallback simple si no hay descriptores
            combined = np.hstack((cv2.resize(img1, (300,300)), cv2.resize(img2, (300,300))))
            _, buf = cv2.imencode('.jpg', combined)
            return buf.tobytes(), 0

        # Matcher BF con Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Dibujar mejores 30 coincidencias
        good_matches = matches[:30]
        match_img = cv2.drawMatches(
            roi1, kp1, roi2, kp2, good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0), singlePointColor=(0, 0, 255)
        )

        _, buffer = cv2.imencode('.jpg', match_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes(), len(good_matches)


    # ==========================================
    # . ENCRIPTACIÓN DEL MORRO Y PATRÓN
    # ==========================================
    # ==========================================
    # OTROS MÉTODOS
    # ==========================================
    def check_spoofing(self, img_bytes: bytes) -> dict:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"is_live_animal": False, "confidence_real": 0.0}
        muzzle_crop, _ = self.crop_muzzle_roi(img)
        spoof_det = self._get_spoof_detector()
        return spoof_det.check_spoofing(muzzle_crop)

    def encrypt_data(self, data_bytes: bytes) -> str:
        encrypted = self.cipher.encrypt(data_bytes)
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_data(self, encrypted_str: str) -> bytes:
        raw_encrypted = base64.b64decode(encrypted_str.encode('utf-8'))
        return self.cipher.decrypt(raw_encrypted)

    def process_tag_ocr(self, tag_img_bytes: bytes) -> str:
        nparr = np.frombuffer(tag_img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        ocr = self._get_ocr()
        results = ocr.readtext(img, detail=0)
        return "".join(results).replace(" ", "").upper()
    
    
    # ==========================================
    #   . OCR DE CARAVANA
    # ==========================================
    def process_tag_ocr(self, tag_img_bytes: bytes) -> str:
        """
        Extrae el texto/número de la imagen de la caravana.
        """
        nparr = np.frombuffer(tag_img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ""

        ocr = self._get_ocr()
        results = ocr.readtext(img, detail=0)
        extracted_text = "".join(results).replace(" ", "").upper()
        return extracted_text
    
    def get_annotated_muzzle(self, image_bytes: bytes):
        """
        Recorta la región nasolabial (ROI), calcula las minucias exactamente sobre
        ese recorte y devuelve la imagen del morro anotada.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagen no válida.")

        # 1. Obtener el recorte del morro (ROI)
        muzzle_roi, (x, y, w, h) = self.crop_muzzle_roi(img)

        # 2. Preprocesar en escala de grises + CLAHE para resaltar rugosidades
        gray = cv2.cvtColor(muzzle_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Detectar minucias / Keypoints directamente sobre la ROI
        orb = cv2.ORB_create(nfeatures=400)
        keypoints = orb.detect(enhanced, None)

        # 4. Dibujar los puntos clave directamente sobre la imagen recortada (muzzle_roi)
        # Dibujar borde verde delimitador alrededor del recorte
        cv2.rectangle(muzzle_roi, (0, 0), (w - 1, h - 1), (0, 255, 0), 3)

        # Dibujar minucias (círculos rojos con punto central)
        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(muzzle_roi, pt, 5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(muzzle_roi, pt, 1, (0, 255, 0), -1, cv2.LINE_AA)

        # 5. Codificar de vuelta a JPEG
        _, buffer = cv2.imencode('.jpg', muzzle_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        bbox_dict = {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}
        return buffer.tobytes(), bbox_dict, len(keypoints)
