import numpy as np
import faiss
from .base import BancoVetorial


class AdaptadorFAISS(BancoVetorial):
    def __init__(self, dimensao_vetor=2048):
        # IndexFlatIP = Inner Product (Produto Interno), que é igual ao Cosseno se os vetores forem normalizados
        self.index = faiss.IndexFlatIP(dimensao_vetor)
        self.ids_map = {} # FAISS usa inteiros como ID, precisamos mapear para nomes
        self.contador = 0

    def adicionar(self, id_item, vetor, metadados):
        # FAISS espera array float32
        vetor_f32 = np.array([vetor], dtype='float32')
        self.index.add(vetor_f32)
        
        # Mapeia o índice numérico do FAISS para o nosso ID (nome do arquivo)
        self.ids_map[self.contador] = id_item
        self.contador += 1

    def buscar(self, vetor_query, top_k):
        vetor_f32 = np.array([vetor_query], dtype='float32')
        scores, indices = self.index.search(vetor_f32, top_k)
        
        retorno = []
        for i in range(top_k):
            idx_faiss = indices[0][i]
            if idx_faiss != -1: # Se encontrou algo
                retorno.append({
                    'id': self.ids_map.get(idx_faiss, "Desconhecido"),
                    'similaridade': scores[0][i], # FAISS IP já retorna a similaridade direta
                    'distancia': 1 - scores[0][i]
                })
        return retorno
