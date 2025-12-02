"""Banco de dados vetorial usando ChromaDB."""
import chromadb
from chromadb.config import Settings
from calculador_similaridade import calcular_distancia_euclidiana


class BancoVetorial:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./banco_musicas")
        
        try:
            self.colecao = self.client.get_collection(name="musicas")
        except:
            self.colecao = self.client.create_collection(name="musicas")
        
        # Carrega cache do banco existente
        self._carregar_cache()
    
    def _carregar_cache(self):
        """Carrega músicas do ChromaDB para o cache."""
        self.todas_musicas = []
        if self.colecao.count() > 0:
            resultados = self.colecao.get(include=['embeddings', 'metadatas'])
            for i in range(len(resultados['ids'])):
                self.todas_musicas.append({
                    'nome': resultados['metadatas'][i]['nome'],
                    'genero': resultados['metadatas'][i]['genero'],
                    'features': resultados['embeddings'][i]
                })
    
    def adicionar(self, nome, genero, features):
        import time
        id_unico = f"{int(time.time()*1000)}_{nome}"
        
        # Converte para lista se necessário
        if hasattr(features, 'tolist'):
            features = features.tolist()
        
        self.colecao.add(
            ids=[id_unico],
            embeddings=[features],
            metadatas=[{"nome": nome, "genero": genero}]
        )
        
        # Adiciona ao cache
        self.todas_musicas.append({
            'nome': nome,
            'genero': genero,
            'features': features
        })
    
    def buscar_similares(self, features, k=5, mostrar_calculos=True):
        """Busca músicas similares usando ChromaDB."""
        if self.colecao.count() == 0:
            return []
        
        # Converte para lista se necessário
        if hasattr(features, 'tolist'):
            features = features.tolist()
        
        resultados = self.colecao.query(
            query_embeddings=[features],
            n_results=k
        )
        
        vizinhos = []
        for i in range(len(resultados['distances'][0])):
            vizinhos.append({
                'nome': resultados['metadatas'][0][i]['nome'],
                'genero': resultados['metadatas'][0][i]['genero'],
                'distancia': resultados['distances'][0][i]
            })
        
        return vizinhos
    
    def buscar_manual(self, features, k=5):
        """Busca manual calculando distâncias passo a passo."""
        if len(self.todas_musicas) == 0:
            return []
        
        # Converte para lista se necessário
        if hasattr(features, 'tolist'):
            features = features.tolist()
        
        print("\n--- CÁLCULO DE DISTÂNCIAS ---")
        
        # Calcula distância para cada música
        distancias = []
        for musica in self.todas_musicas:
            dist = calcular_distancia_euclidiana(features, musica['features'])
            distancias.append({
                'nome': musica['nome'],
                'genero': musica['genero'],
                'distancia': dist
            })
            print(f"{musica['nome']}: {dist:.4f}")
        
        # Ordena por distância (menor primeiro) usando bubble sort
        for i in range(len(distancias)):
            for j in range(i + 1, len(distancias)):
                if distancias[i]['distancia'] > distancias[j]['distancia']:
                    distancias[i], distancias[j] = distancias[j], distancias[i]
        
        # Retorna os k mais próximos
        return distancias[:k]
    
    def total(self):
        return self.colecao.count()
