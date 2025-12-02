import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


class VisualizadorDeComparacao:
    """
    Classe para criar visualizações detalhadas das comparações de imagens.
    """

    @staticmethod
    def plotar_comparacao_detalhada(
        img1_path: str,
        img2_path: str,
        vetor1: np.ndarray,
        vetor2: np.ndarray,
        similaridade: float,
        salvar_em: str = None,
    ):
        """
        Cria um plot detalhado comparando duas imagens.
        """
        # Configurar o estilo
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))

        # Grid de subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. IMAGENS ORIGINAIS
        ax1 = fig.add_subplot(gs[0, 0])
        img1 = Image.open(img1_path)
        ax1.imshow(img1)
        ax1.set_title(
            f"Imagem 1\n{os.path.basename(img1_path)}", fontsize=12, fontweight="bold"
        )
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        img2 = Image.open(img2_path)
        ax2.imshow(img2)
        ax2.set_title(
            f"Imagem 2\n{os.path.basename(img2_path)}", fontsize=12, fontweight="bold"
        )
        ax2.axis("off")

        # 2. MEDIDOR DE SIMILARIDADE
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis("off")

        # Criar medidor visual
        categorias = [
            "Muito\nDiferentes",
            "Diferentes",
            "Similares",
            "Muito\nSimilares",
            "Idênticas",
        ]
        cores = ["#d32f2f", "#f57c00", "#fbc02d", "#689f38", "#388e3c"]
        limites = [0, 0.3, 0.5, 0.7, 0.9, 1.0]

        # Determinar categoria
        categoria_idx = 0
        for i in range(len(limites) - 1):
            if limites[i] <= similaridade < limites[i + 1]:
                categoria_idx = i
                break

        # Texto principal
        ax3.text(
            0.5,
            0.7,
            "SIMILARIDADE",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.5,
            0.45,
            f"{similaridade:.4f}",
            ha="center",
            va="center",
            fontsize=40,
            fontweight="bold",
            color=cores[categoria_idx],
            transform=ax3.transAxes,
        )
        ax3.text(
            0.5,
            0.25,
            categorias[categoria_idx],
            ha="center",
            va="center",
            fontsize=14,
            color=cores[categoria_idx],
            fontweight="bold",
            transform=ax3.transAxes,
        )

        # Barra de progresso
        rect_y = 0.05
        rect_height = 0.08
        for i, cor in enumerate(cores):
            rect_x = i * 0.18 + 0.1
            rect = plt.Rectangle(
                (rect_x, rect_y),
                0.16,
                rect_height,
                facecolor=cor,
                alpha=0.3,
                transform=ax3.transAxes,
            )
            ax3.add_patch(rect)

            # Destacar a categoria atual
            if i == categoria_idx:
                rect_destaque = plt.Rectangle(
                    (rect_x - 0.01, rect_y - 0.01),
                    0.18,
                    rect_height + 0.02,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=3,
                    transform=ax3.transAxes,
                )
                ax3.add_patch(rect_destaque)

        # 3. DISTRIBUIÇÃO DOS VETORES (primeiros 100 valores)
        ax4 = fig.add_subplot(gs[1, :])
        x = np.arange(100)
        ax4.plot(
            x,
            vetor1[:100],
            label="Vetor Imagem 1",
            color="#1976d2",
            linewidth=2,
            alpha=0.7,
        )
        ax4.plot(
            x,
            vetor2[:100],
            label="Vetor Imagem 2",
            color="#d32f2f",
            linewidth=2,
            alpha=0.7,
        )
        ax4.set_xlabel("Dimensão do Vetor", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Valor", fontsize=11, fontweight="bold")
        ax4.set_title(
            "Primeiras 100 Dimensões dos Vetores de Features",
            fontsize=12,
            fontweight="bold",
        )
        ax4.legend(loc="upper right", fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 4. ESTATÍSTICAS DOS VETORES
        ax5 = fig.add_subplot(gs[2, 0])
        estatisticas = [
            ["Métrica", "Imagem 1", "Imagem 2"],
            ["Média", f"{np.mean(vetor1):.4f}", f"{np.mean(vetor2):.4f}"],
            ["Desvio Padrão", f"{np.std(vetor1):.4f}", f"{np.std(vetor2):.4f}"],
            ["Máximo", f"{np.max(vetor1):.4f}", f"{np.max(vetor2):.4f}"],
            ["Mínimo", f"{np.min(vetor1):.4f}", f"{np.min(vetor2):.4f}"],
            [
                "Norma L2",
                f"{np.linalg.norm(vetor1):.4f}",
                f"{np.linalg.norm(vetor2):.4f}",
            ],
        ]

        ax5.axis("tight")
        ax5.axis("off")
        table = ax5.table(
            cellText=estatisticas,
            cellLoc="center",
            loc="center",
            colWidths=[0.4, 0.3, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Estilizar header
        for i in range(3):
            table[(0, i)].set_facecolor("#1976d2")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax5.set_title(
            "Estatísticas dos Vetores", fontsize=12, fontweight="bold", pad=10
        )

        # 5. HISTOGRAMA DE DISTRIBUIÇÃO
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(
            vetor1,
            bins=50,
            alpha=0.6,
            color="#1976d2",
            label="Imagem 1",
            edgecolor="black",
        )
        ax6.hist(
            vetor2,
            bins=50,
            alpha=0.6,
            color="#d32f2f",
            label="Imagem 2",
            edgecolor="black",
        )
        ax6.set_xlabel("Valor", fontsize=10, fontweight="bold")
        ax6.set_ylabel("Frequência", fontsize=10, fontweight="bold")
        ax6.set_title("Distribuição dos Valores", fontsize=12, fontweight="bold")
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis="y")

        # 6. CÁLCULO PASSO A PASSO
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        dot_product = np.dot(vetor1, vetor2)
        norm_a = np.linalg.norm(vetor1)
        norm_b = np.linalg.norm(vetor2)

        passos = [
            "CÁLCULO MATEMÁTICO\n",
            "1. Produto Escalar (A · B):",
            f"   {dot_product:.6f}",
            "",
            "2. Norma de A (||A||):",
            f"   {norm_a:.6f}",
            "",
            "3. Norma de B (||B||):",
            f"   {norm_b:.6f}",
            "",
            "4. Fórmula Final:",
            "   cos(θ) = (A · B) / (||A|| × ||B||)",
            f"   cos(θ) = {similaridade:.6f}",
            "",
            f"5. Ângulo entre vetores:",
            f"   θ ≈ {np.arccos(np.clip(similaridade, -1, 1)) * 180 / np.pi:.2f}°",
        ]

        texto_completo = "\n".join(passos)
        ax7.text(
            0.1,
            0.95,
            texto_completo,
            transform=ax7.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        # Título principal
        fig.suptitle(
            "Análise Detalhada de Similaridade de Imagens",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        # Salvar ou mostrar
        if salvar_em:
            plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
            print(f"\n✅ Gráfico salvo em: {salvar_em}")
        else:
            # Auto-salvar com nome padrão se não especificado
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"comparacao_{timestamp}.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
            print(f"\n✅ Gráfico salvo automaticamente em: {nome_arquivo}")

        plt.close()

    @staticmethod
    def plotar_heatmap_comparacao(
        vetor1: np.ndarray,
        vetor2: np.ndarray,
        img1_name: str,
        img2_name: str,
        salvar_em: str = None,
    ):
        """
        Cria um heatmap comparando os vetores.
        """
        # Reshape para matriz 32x64 (2048 = 32*64)
        v1_matrix = vetor1.reshape(32, 64)
        v2_matrix = vetor2.reshape(32, 64)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Heatmap Imagem 1
        sns.heatmap(v1_matrix, cmap="viridis", ax=axes[0], cbar_kws={"label": "Valor"})
        axes[0].set_title(
            f"Heatmap do Vetor\n{img1_name}", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Dimensão (64)", fontsize=10)
        axes[0].set_ylabel("Dimensão (32)", fontsize=10)

        # Heatmap Imagem 2
        sns.heatmap(v2_matrix, cmap="viridis", ax=axes[1], cbar_kws={"label": "Valor"})
        axes[1].set_title(
            f"Heatmap do Vetor\n{img2_name}", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Dimensão (64)", fontsize=10)
        axes[1].set_ylabel("Dimensão (32)", fontsize=10)

        plt.suptitle(
            "Representação Visual dos Vetores de Features",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if salvar_em:
            plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
            print(f"✅ Heatmap salvo em: {salvar_em}")
        else:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"heatmap_{timestamp}.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
            print(f"✅ Heatmap salvo automaticamente em: {nome_arquivo}")

        plt.close()

    @staticmethod
    def plotar_resultados_busca(
        imagem_query: str, resultados: list, salvar_em: str = None
    ):
        """
        Visualiza os resultados de uma busca por similaridade.
        """
        n_resultados = len(resultados)
        fig, axes = plt.subplots(
            1, n_resultados + 1, figsize=(4 * (n_resultados + 1), 5)
        )

        if n_resultados == 0:
            return

        # Imagem de query
        img_query = Image.open(imagem_query)
        axes[0].imshow(img_query)
        axes[0].set_title(
            f"🔍 BUSCA\n{os.path.basename(imagem_query)}",
            fontsize=12,
            fontweight="bold",
            color="#d32f2f",
        )
        axes[0].axis("off")
        axes[0].set_facecolor("#ffebee")

        # Resultados
        for i, res in enumerate(resultados):
            try:
                # Tentar encontrar a imagem
                img_path = res.get("path", res["id"])
                if not os.path.exists(img_path):
                    # Procurar na pasta fotos
                    img_path = os.path.join("fotos", res["id"])

                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[i + 1].imshow(img)

                similaridade = res["similaridade"]
                cor = (
                    "#388e3c"
                    if similaridade > 0.7
                    else "#f57c00"
                    if similaridade > 0.5
                    else "#d32f2f"
                )

                axes[i + 1].set_title(
                    f"#{i + 1} - {res['id']}\nSimilaridade: {similaridade:.4f}",
                    fontsize=11,
                    fontweight="bold",
                    color=cor,
                )
                axes[i + 1].axis("off")

                # Cor de fundo baseada na similaridade
                if similaridade > 0.7:
                    axes[i + 1].set_facecolor("#e8f5e9")
                elif similaridade > 0.5:
                    axes[i + 1].set_facecolor("#fff3e0")
                else:
                    axes[i + 1].set_facecolor("#ffebee")

            except Exception as e:
                axes[i + 1].text(
                    0.5,
                    0.5,
                    f"Erro ao\ncarregar\nimagem",
                    ha="center",
                    va="center",
                    transform=axes[i + 1].transAxes,
                )
                axes[i + 1].axis("off")

        plt.suptitle(
            "Resultados da Busca por Similaridade", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if salvar_em:
            plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
            print(f"\n✅ Resultados salvos em: {salvar_em}")
        else:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"busca_{timestamp}.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches="tight")
            print(f"\n✅ Resultados salvos automaticamente em: {nome_arquivo}")

        plt.close()
