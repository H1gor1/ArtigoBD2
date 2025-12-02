"""Cálculo de similaridade entre vetores."""
import math


def calcular_distancia_euclidiana(vetor1, vetor2):
    """
    Calcula distância euclidiana PONDERADA entre dois vetores.
    Quanto MENOR a distância, MAIS similares são os vetores.
    
    Pesos aplicados:
    - MFCCs (0-12): peso 3.0 (muito importantes para timbre)
    - Spectral centroid (13): peso 2.0 (brilho do som)
    - Spectral rolloff (14): peso 1.5 (conteúdo de alta frequência)
    - Zero crossing rate (15): peso 1.5 (percussão/ruído)
    - Chroma (16-27): peso 4.0 (harmonia/tonalidade)
    - Tempo (28): peso 2.5 (ritmo)
    """
    if len(vetor1) != len(vetor2):
        raise ValueError("Vetores devem ter o mesmo tamanho")
    
    # Define pesos para cada feature (rebalanceados)
    pesos = []
    # MFCCs (13 features): peso 8.0 (CRÍTICO para timbre/textura)
    pesos.extend([8.0] * 13)
    # Spectral centroid: peso 4.0 (brilho do som)
    pesos.append(4.0)
    # Spectral rolloff: peso 3.5 (conteúdo de frequência)
    pesos.append(3.5)
    # Zero crossing rate: peso 5.0 (percussão/ruído)
    pesos.append(5.0)
    # Chroma (12 features): peso 2.0 (reduzido - estava causando confusão)
    pesos.extend([2.0] * 12)
    # Tempo: peso 3.5
    pesos.append(3.5)
    
    # Ajusta se o vetor for menor (retrocompatibilidade)
    if len(vetor1) < len(pesos):
        pesos = pesos[:len(vetor1)]
    elif len(vetor1) > len(pesos):
        pesos.extend([1.0] * (len(vetor1) - len(pesos)))
    
    # Soma ponderada das diferenças ao quadrado
    soma = 0
    for i in range(len(vetor1)):
        diferenca = vetor1[i] - vetor2[i]
        soma += pesos[i] * diferenca * diferenca
    
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
