from abc import ABC, abstractmethod
import numpy as np


class BancoVetorial(ABC):
    """
    Contrato que obriga qualquer banco de dados a ter os métodos adicionar e buscar.
    """
    @abstractmethod
    def adicionar(self, id_item: str, vetor: np.ndarray, metadados: dict):
        pass

    @abstractmethod
    def buscar(self, vetor_query: np.ndarray, top_k: int):
        pass
