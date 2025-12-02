"""Classificador de gêneros musicais."""

import os

from banco_vetorial import BancoVetorial
from extrator_features import ExtratorFeatures


class ClassificadorMusical:
    def __init__(self):
        self.extrator = ExtratorFeatures()
        self.banco = BancoVetorial()

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
        """Classifica uma música."""
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

        # Votação híbrida: combina contagem com peso por distância
        print("\n--- VOTAÇÃO HÍBRIDA ---")
        votos = {}
        peso_total = {}

        for i, v in enumerate(vizinhos):
            genero = v["genero"]
            # Peso inversamente proporcional à distância
            peso = 1.0 / (v["distancia"] + 1.0)

            votos[genero] = votos.get(genero, 0) + 1
            peso_total[genero] = peso_total.get(genero, 0) + peso

        # Score híbrido: 50% peso por distância + 50% peso por votação
        score_final = {}
        max_votos = max(votos.values())
        sum_peso = sum(peso_total.values())

        for genero in votos.keys():
            peso_norm = peso_total[genero] / sum_peso
            voto_norm = votos[genero] / max_votos
            score_final[genero] = 0.5 * peso_norm + 0.5 * voto_norm
            print(
                f"{genero}: {votos[genero]} votos, peso: {peso_total[genero]:.4f}, score: {score_final[genero]:.4f}"
            )

        # Escolhe o gênero com maior score final
        genero_final = max(score_final, key=score_final.get)
        confianca = (score_final[genero_final] / sum(score_final.values())) * 100

        print(f"\n{'=' * 40}")
        print(f"RESULTADO: {genero_final}")
        print(f"Confiança: {confianca:.1f}%")
        print(f"{'=' * 40}")

        return genero_final
