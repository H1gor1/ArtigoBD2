import chromadb
import numpy as np

from .base import BancoVetorial


class AdaptadorChromaDB(BancoVetorial):
    def __init__(self):
        self.client = (
            chromadb.PersistentClient()
        )  # Memória volátil para teste (use PersistentClient para salvar)
        self.collection = self.client.get_or_create_collection(
            name="imagens", metadata={"hnsw:space": "cosine"}
        )

    def adicionar(self, id_item, vetor, metadados):
        self.collection.add(
            ids=[id_item], embeddings=[vetor.tolist()], metadatas=[metadados]
        )

    def buscar(self, vetor_query, top_k):
        resultados = self.collection.query(
            query_embeddings=[vetor_query.tolist()], n_results=top_k
        )
        # Simplificando a saída do Chroma
        retorno = []
        for i in range(len(resultados["ids"][0])):
            retorno.append(
                {
                    "id": resultados["ids"][0][i],
                    "distancia": resultados["distances"][0][i],
                    # Chroma retorna distância, similaridade é 1 - distancia
                    "similaridade": 1 - resultados["distances"][0][i],
                }
            )
        return retorno
