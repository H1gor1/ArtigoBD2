"""Classificador de gêneros musicais."""

import os

from banco_vetorial import BancoVetorial
from extrator_features import ExtratorFeatures
from visualizador import Visualizador


class ClassificadorMusical:
    def __init__(self):
        self.extrator = ExtratorFeatures()
        self.banco = BancoVetorial()
        self.visualizador = Visualizador()

    def adicionar_musica(self, caminho, genero):
        """Adiciona uma música ao banco."""
        if not os.path.exists(caminho):
            print(f"Erro: Arquivo não encontrado: {caminho}")
            return

        print(f"\nProcessando: {caminho}")
        features = self.extrator.extrair_todas_features(caminho)
        nome = os.path.basename(caminho)
        self.banco.adicionar(nome, genero, features)
        print(f"✓ Adicionada: {nome} ({genero})")

    def classificar_musica(self, caminho, k=5, mostrar_calculos=True):
        """Classifica uma música e retorna os resultados."""
        if self.banco.total() == 0:
            print("Erro: Adicione músicas primeiro!")
            return None

        if not os.path.exists(caminho):
            print(f"Erro: Arquivo não encontrado: {caminho}")
            return None

        print(f"\nClassificando: {caminho}")
        features = self.extrator.extrair_todas_features(caminho)

        # Busca manual (mostra cálculos)
        if mostrar_calculos:
            vizinhos = self.banco.buscar_manual(features, k)
        else:
            vizinhos = self.banco.buscar_similares(features, k)

        # Mostra vizinhos
        print(f"\n--- {k} VIZINHOS MAIS PRÓXIMOS ---")
        for i, v in enumerate(vizinhos, 1):
            print(f"{i}. {v['nome']} - {v['genero']} (dist: {v['distancia']:.4f})")

        # Votação por maioria simples
        print("\n--- VOTAÇÃO ---")
        votos = {}

        for v in vizinhos:
            genero = v["genero"]
            votos[genero] = votos.get(genero, 0) + 1

        for genero, contagem in votos.items():
            print(f"{genero}: {contagem} votos")

        # Escolhe o gênero com mais votos
        genero_final = max(votos, key=votos.get)
        confianca = (votos[genero_final] / k) * 100

        print(f"\n{'=' * 40}")
        print(f"RESULTADO: {genero_final}")
        print(f"Confiança: {confianca:.1f}%")
        print(f"{'=' * 40}")

        # Gera visualização automaticamente
        nome_musica = os.path.basename(caminho)
        print("\nGerando visualização...")

        fig = self.visualizador.plotar_resultados(
            nome_musica=nome_musica,
            vizinhos=vizinhos,
            votos=votos,
            genero_final=genero_final,
            confianca=confianca,
            k=k,
        )

        # Salva o gráfico
        nome_arquivo = os.path.splitext(nome_musica)[0]
        caminho_saida = f"classificacao_{nome_arquivo}.png"
        self.visualizador.salvar_grafico(fig, caminho_saida)

        # Retorna dicionário com todos os dados
        return {
            "genero": genero_final,
            "confianca": confianca,
            "vizinhos": vizinhos,
            "votos": votos,
            "k": k,
            "nome_musica": nome_musica,
            "grafico": caminho_saida,
        }
