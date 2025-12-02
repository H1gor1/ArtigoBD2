import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image


class ExtratorDeFeatures:
    """
    Responsável APENAS por transformar uma imagem em uma lista de números.
    """
    def __init__(self):
        # Usamos ResNet50 (uma rede neural pré-treinada)
        # weights='DEFAULT' usa os pesos mais atuais
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Removemos a última camada (que classifica se é gato, cachorro, etc)
        # Queremos apenas os números anteriores a essa decisão (o embedding)
        self.modelo = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.modelo.eval() # Modo de avaliação (não treino)

        # Regras de transformação para a imagem entrar na rede neural
        self.transformacao = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def gerar_vetor(self, caminho_imagem: str) -> np.ndarray:
        try:
            # 1. Carregar imagem
            img = Image.open(caminho_imagem).convert('RGB')
            # 2. Transformar em tensores (formato que a IA entende)
            img_t = self.transformacao(img).unsqueeze(0)
            
            # 3. Passar pela rede neural
            with torch.no_grad():
                features = self.modelo(img_t)
            
            # 4. Transformar em array simples do numpy (uma lista de números)
            features_np = features.flatten().numpy()
            
            # 5. Normalizar (deixar o vetor com tamanho 1 para facilitar cálculos)
            features_np = features_np / np.linalg.norm(features_np)
            
            return features_np
        except Exception as e:
            print(f"Erro ao processar imagem {caminho_imagem}: {e}")
            return np.zeros(2048) # Retorna vetor vazio em caso de erro
