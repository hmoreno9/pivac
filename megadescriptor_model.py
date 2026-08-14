import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

#Red Siamesa ResNet-18

class MuzzleMegaDescriptor(nn.Module):
    def __init__(self, embedding_dimension: int = 512, pretrained: bool = True):
        super(MuzzleMegaDescriptor, self).__init__()
        # Usamos ResNet-18 como backbone para equilibrio entre velocidad y precisión
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)
        
        # Eliminar la última capa de clasificación (fc)
        self.backbone = nn.Sequential(*list(base_resnet.children())[:-1])
        
        # Capa de proyección para el embedding biométrico
        self.fc_embedding = nn.Linear(base_resnet.fc.in_features, embedding_dimension)
        self.bn_embedding = nn.BatchNorm1d(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embedding = self.fc_embedding(features)
        embedding = self.bn_embedding(embedding)
        
        # Normalización L2 imprescindible para similitud Coseno
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        return embedding / (norm + 1e-10)


class MegaDescriptorWrapper:
    def __init__(self, weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MuzzleMegaDescriptor(embedding_dimension=512, pretrained=True)
        
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()

        # Transformaciones estándar para la entrada a la red
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_embedding(self, cropped_muzzle_cv2: np.ndarray) -> np.ndarray:
        """
        Recibe una imagen recortada (Crop) del morro en formato BGR de OpenCV
        y retorna un vector de 512 Float32 normalizado.
        """
        # Convertir OpenCV BGR -> PIL Image RGB
        img_rgb = Image.fromarray(cropped_muzzle_cv2[:, :, ::-1])
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)
            
        return embedding.cpu().numpy().flatten().astype(np.float32)
