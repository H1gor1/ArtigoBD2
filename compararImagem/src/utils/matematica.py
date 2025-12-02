import numpy as np


class SimilarityCalculator:
    """
    Classe utilitária apenas para demonstrar como o cálculo é feito.
    Na prática, os bancos de dados fazem isso muito rápido, mas aqui vamos
    abrir a 'caixa preta'.
    """

    @staticmethod
    def calcular_similaridade_cosseno(vetor_a, vetor_b):
        print("\n--- INICIANDO CÁLCULO PASSO A PASSO ---")

        # Passo 1: Produto Escalar (Dot Product)
        # Multiplica cada número de A pelo correspondente em B e soma tudo.
        # Se A = [1, 2] e B = [3, 4], dot = (1*3) + (2*4) = 11
        dot_product = np.dot(vetor_a, vetor_b)
        print(f"1. Produto Escalar (Soma das multiplicações): {dot_product:.4f}")

        # Passo 2: Magnitude (Norma/Tamanho do vetor)
        # É o comprimento da seta do vetor. Raiz quadrada da soma dos quadrados.
        norm_a = np.linalg.norm(vetor_a)
        norm_b = np.linalg.norm(vetor_b)
        print(f"2. Tamanho do Vetor A: {norm_a:.4f }")
        print(f"3. Tamanho do Vetor B: {norm_b:.4f}")

        # Passo 3: Divisão final
        # Fórmula: (A . B) / (||A|| * ||B||)
        similaridade = dot_product / (norm_a * norm_b)
        print(f"4. Resultado (Cosseno): {similaridade:.4f}")
        print("---------------------------------------\n")

        return similaridade
