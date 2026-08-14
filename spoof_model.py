import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

class MuzzleSpoofDetector(nn.Module):
    def __init__(self):
        super(MuzzleSpoofDetector, self).__init__()
        # Clasificador binario: 0 = Fake/Screen/Paper, 1 = Real Muzzle Live
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # [Spoof, Real]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SpoofDetectorWrapper:
    def __init__(self, weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuzzleSpoofDetector()
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def check_spoofing(self, cropped_muzzle_bgr: np.ndarray) -> dict:
        """
        Combina detección por modelo profundo + análisis de frecuencia (Fourier)
        para detectar brillo de pantallas/impresiones.
        """
        # 1. Análisis de Frecuencia (FFT) para buscar artefactos de muaré / pantallas
        gray = cv2.cvtColor(cropped_muzzle_bgr, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
        mean_freq = np.mean(magnitude_spectrum)

        # 2. Inferencia del modelo de Anti-Spoofing
        img_rgb = Image.fromarray(cropped_muzzle_bgr[:, :, ::-1])
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        is_real = probs[1] > 0.80 and mean_freq < 170.0

        return {
            "is_live_animal": bool(is_real),
            "confidence_real": float(probs[1]),
            "fft_spectrum_mean": float(mean_freq)
        }
