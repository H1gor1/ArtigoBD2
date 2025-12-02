"""Cálculo de similaridade entre vetores."""
import math


def calcular_distancia_euclidiana(vetor1, vetor2):
    """
    Calcula distância euclidiana padrão entre dois vetores.
    Quanto MENOR a distância, MAIS similares são os vetores.
    
    Fórmula: d = sqrt(sum((a_i - b_i)^2))
    """
    if len(vetor1) != len(vetor2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    
    # Soma das diferenças ao quadrado
    soma = 0
    for i in range(len(vetor1)):
        diferenca = vetor1[i] - vetor2[i]
        soma += diferenca * diferenca
    
    # Raiz quadrada
    distancia = math.sqrt(soma)
    return distancia


def calcular_similaridade_cosseno(vetor1, vetor2):
    """
    Calcula similaridade do cosseno entre dois vetores.
    Retorna valor entre -1 e 1, onde 1 = idênticos.
    
    Fórmula: cos(θ) = (A · B) / (||A|| * ||B||)
    """
    if len(vetor1) != len(vetor2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    
    # Produto escalar (dot product)
    produto = 0
    for i in range(len(vetor1)):
        produto += vetor1[i] * vetor2[i]
    
    # Magnitude do vetor1
    mag1 = 0
    for valor in vetor1:
        mag1 += valor * valor
    mag1 = math.sqrt(mag1)
    
    # Magnitude do vetor2
    mag2 = 0
    for valor in vetor2:
        mag2 += valor * valor
    mag2 = math.sqrt(mag2)
    
    # Evita divisão por zero
    if mag1 == 0 or mag2 == 0:
        return 0
    
    similaridade = produto / (mag1 * mag2)
    return similaridade
