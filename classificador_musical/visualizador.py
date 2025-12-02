"""Visualização dos resultados da classificação musical."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class Visualizador:
    def __init__(self):
        """Inicializa o visualizador."""
        plt.style.use("ggplot")

    def plotar_resultados(
        self, nome_musica, vizinhos, votos, genero_final, confianca, k
    ):
        """
        Cria uma visualização completa dos resultados da classificação.

        Args:
            nome_musica: Nome do arquivo de música
            vizinhos: Lista de dicionários com 'nome', 'genero', 'distancia'
            votos: Dicionário com contagem de votos por gênero
            genero_final: Gênero classificado
            confianca: Percentual de confiança
            k: Número de vizinhos usados
        """
        # Cria figura com 3 subplots
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"Classificação Musical: {nome_musica}", fontsize=16, fontweight="bold"
        )

        # 1. Gráfico de distâncias dos vizinhos
        ax1 = plt.subplot(2, 2, 1)
        self._plotar_distancias(ax1, vizinhos)

        # 2. Gráfico de votação por gênero
        ax2 = plt.subplot(2, 2, 2)
        self._plotar_votacao(ax2, votos, genero_final)

        # 3. Distribuição dos vizinhos por gênero
        ax3 = plt.subplot(2, 2, 3)
        self._plotar_distribuicao_vizinhos(ax3, vizinhos)

        # 4. Resultado final
        ax4 = plt.subplot(2, 2, 4)
        self._plotar_resultado_final(ax4, genero_final, confianca, k)

        plt.tight_layout()
        return fig

    def _plotar_distancias(self, ax, vizinhos):
        """Plota as distâncias dos K vizinhos mais próximos."""
        nomes = [f"{i + 1}. {v['nome'][:15]}" for i, v in enumerate(vizinhos)]
        distancias = [v["distancia"] for v in vizinhos]
        generos = [v["genero"] for v in vizinhos]

        # Cores por gênero
        cores_genero = self._gerar_cores_generos(generos)
        cores = [cores_genero[g] for g in generos]

        bars = ax.barh(nomes, distancias, color=cores, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Distância Euclidiana", fontweight="bold")
        ax.set_ylabel("Vizinhos", fontweight="bold")
        ax.set_title("K Vizinhos Mais Próximos", fontweight="bold")
        ax.invert_yaxis()

        # Adiciona valores nas barras
        for i, (bar, dist) in enumerate(zip(bars, distancias)):
            ax.text(dist + 0.5, i, f"{dist:.2f}", va="center")

    def _plotar_votacao(self, ax, votos, genero_final):
        """Plota a votação por gênero."""
        generos = list(votos.keys())
        contagens = list(votos.values())

        cores_genero = self._gerar_cores_generos(generos)
        cores = [cores_genero[g] for g in generos]

        # Cria barras individualmente para controlar alpha
        bars = []
        for i, (genero, contagem) in enumerate(zip(generos, contagens)):
            alpha = 1.0 if genero == genero_final else 0.6
            bar = ax.bar(
                i, contagem, color=cores[i], alpha=alpha, edgecolor="black", linewidth=2
            )
            bars.append(bar)

        # Adiciona estrela no vencedor
        idx_vencedor = generos.index(genero_final)
        ax.text(
            idx_vencedor,
            contagens[idx_vencedor] + 0.3,
            "★",
            ha="center",
            fontsize=30,
            color="gold",
            fontweight="bold",
        )

        ax.set_xlabel("Gênero", fontweight="bold")
        ax.set_ylabel("Número de Votos", fontweight="bold")
        ax.set_title("Votação por Gênero", fontweight="bold")
        ax.set_ylim(0, max(contagens) + 1)
        ax.set_xticks(range(len(generos)))
        ax.set_xticklabels(generos)

        # Adiciona valores nas barras
        for i, contagem in enumerate(contagens):
            ax.text(i, contagem + 0.1, str(contagem), ha="center", fontweight="bold")

    def _plotar_distribuicao_vizinhos(self, ax, vizinhos):
        """Plota a distribuição dos vizinhos por gênero em pizza."""
        generos = [v["genero"] for v in vizinhos]
        unique_generos = list(set(generos))
        contagens = [generos.count(g) for g in unique_generos]

        cores_genero = self._gerar_cores_generos(unique_generos)
        cores = [cores_genero[g] for g in unique_generos]

        wedges, texts, autotexts = ax.pie(
            contagens,
            labels=unique_generos,
            colors=cores,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontweight": "bold"},
        )

        # Destaca percentuais
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")

        ax.set_title("Distribuição dos Vizinhos", fontweight="bold")

    def _plotar_resultado_final(self, ax, genero_final, confianca, k):
        """Plota o resultado final da classificação."""
        ax.axis("off")

        # Box para o resultado
        box_width = 0.8
        box_height = 0.6
        box_x = 0.1
        box_y = 0.2

        cores_genero = self._gerar_cores_generos([genero_final])
        cor = cores_genero[genero_final]

        rect = Rectangle(
            (box_x, box_y),
            box_width,
            box_height,
            facecolor=cor,
            edgecolor="black",
            linewidth=3,
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Texto principal
        ax.text(
            0.5,
            0.65,
            "RESULTADO",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="black",
        )

        ax.text(
            0.5,
            0.5,
            genero_final.upper(),
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
            color=cor,
        )

        ax.text(
            0.5,
            0.35,
            f"Confiança: {confianca:.1f}%",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="black",
        )

        # Informações adicionais
        ax.text(
            0.5,
            0.1,
            f"K = {k} vizinhos",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            color="gray",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Classificação Final", fontweight="bold", pad=20)

    def _gerar_cores_generos(self, generos):
        """Gera um mapa de cores consistente para os gêneros."""
        generos_unicos = list(set(generos))

        # Paleta de cores predefinida
        cores_padrao = {
            "rock": "#e74c3c",  # vermelho
            "pop": "#3498db",  # azul
            "jazz": "#9b59b6",  # roxo
            "classical": "#2ecc71",  # verde
            "eletronica": "#f39c12",  # laranja
            "hip-hop": "#1abc9c",  # turquesa
            "reggae": "#d35400",  # laranja escuro
            "blues": "#34495e",  # cinza azulado
            "metal": "#7f8c8d",  # cinza
            "folk": "#16a085",  # verde água
        }

        # Gera cores para gêneros não mapeados
        cores_extras = plt.cm.Set3(np.linspace(0, 1, len(generos_unicos)))

        cores_mapa = {}
        for i, genero in enumerate(generos_unicos):
            genero_lower = genero.lower()
            if genero_lower in cores_padrao:
                cores_mapa[genero] = cores_padrao[genero_lower]
            else:
                cores_mapa[genero] = cores_extras[i]

        return cores_mapa

    def salvar_grafico(self, fig, caminho_saida="resultado_classificacao.png"):
        """Salva o gráfico em arquivo."""
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico salvo em: {caminho_saida}")

    def mostrar_grafico(self, fig):
        """Exibe o gráfico na tela."""
        plt.show()
