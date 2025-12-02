import numpy as np
import os
from .models import ExtratorDeFeatures
from .database import AdaptadorChromaDB, AdaptadorFAISS
from .utils import SimilarityCalculator, VisualizadorDeComparacao


class ComparadorDeImagens:
    def __init__(self, usar_banco: str = 'chroma'):
        self.extrator = ExtratorDeFeatures()
        
        # Aqui decidimos qual "motor" de banco de dados usar
        if usar_banco.lower() == 'faiss':
            print(">> Inicializando com FAISS")
            self.db = AdaptadorFAISS()
        else:
            print(">> Inicializando com ChromaDB")
            self.db = AdaptadorChromaDB()

    def indexar_imagem(self, caminho: str):
        """Adiciona uma imagem ao banco de dados vetorial"""
        print(f"Indexando: {caminho}...")
        vetor = self.extrator.gerar_vetor(caminho)
        # O ID será o nome do arquivo
        nome_arquivo = os.path.basename(caminho)
        self.db.adicionar(id_item=nome_arquivo, vetor=vetor, metadados={"path": caminho})

    def indexar_pasta(self, pasta: str):
        """Indexa todas as imagens de uma pasta"""
        extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(extensoes):
                caminho_completo = os.path.join(pasta, arquivo)
                self.indexar_imagem(caminho_completo)

    def buscar_similares(self, caminho_query: str, top_k: int = 3, 
                        visualizar: bool = False, salvar_plot: str = None):
        """Busca imagens similares no banco de dados"""
        print(f"\nBuscando {top_k} imagens similares para: {caminho_query}")
        vetor_query = self.extrator.gerar_vetor(caminho_query)
        resultados = self.db.buscar(vetor_query, top_k=top_k)
        
        # Gerar visualização se solicitado
        if visualizar or salvar_plot:
            VisualizadorDeComparacao.plotar_resultados_busca(
                caminho_query, resultados, salvar_em=salvar_plot
            )
        
        return resultados

    def comparar_duas_imagens(self, img1: str, img2: str, mostrar_detalhes: bool = False, 
                             visualizar: bool = False, salvar_plot: str = None):
        """Compara duas imagens diretamente e retorna a similaridade"""
        print(f"\nComparando: {img1} vs {img2}")
        v1 = self.extrator.gerar_vetor(img1)
        v2 = self.extrator.gerar_vetor(img2)
        
        if mostrar_detalhes:
            similaridade = SimilarityCalculator.calcular_similaridade_cosseno(v1, v2)
        else:
            similaridade = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            print(f"Similaridade: {similaridade:.4f}")
        
        # Gerar visualização se solicitado
        if visualizar or salvar_plot:
            VisualizadorDeComparacao.plotar_comparacao_detalhada(
                img1, img2, v1, v2, similaridade, salvar_em=salvar_plot
            )
        
        return similaridade
