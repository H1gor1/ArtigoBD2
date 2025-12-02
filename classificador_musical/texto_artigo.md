# Metodologia e Técnicas Utilizadas no Classificador Musical

## 1. Extração de Características (Feature Extraction)

O sistema implementa uma pipeline de extração de características de áudio utilizando a biblioteca Librosa (McFee et al., 2015), processando os primeiros 230 segundos de cada arquivo de áudio. As características extraídas incluem:

### 1.1 Mel-Frequency Cepstral Coefficients (MFCCs)

Os MFCCs são amplamente reconhecidos como a representação mais efetiva para características timbrísticas de áudio (Logan, 2000). O sistema extrai 13 coeficientes MFCCs, que capturam a forma espectral do envelope do sinal de áudio. Estudos demonstram que MFCCs são particularmente eficazes para distinguir instrumentos e texturas sonoras (Tzanetakis & Cook, 2002), sendo fundamentais para a discriminação entre gêneros musicais.

### 1.2 Características Espectrais

#### Centróide Espectral (Spectral Centroid)
O centróide espectral representa o "centro de massa" do espectro de frequências e está fortemente correlacionado com a percepção de brilho sonoro (Grey, 1977). Valores mais altos indicam sons mais brilhantes, o que é característico de diferentes gêneros musicais.

#### Spectral Rolloff
Esta métrica indica a frequência abaixo da qual se concentra 85% da energia espectral, sendo útil para distinguir entre sons harmônicos e ruidosos (Scheirer, 1998). Gêneros com maior conteúdo de alta frequência apresentam valores de rolloff mais elevados.

#### Zero Crossing Rate (ZCR)
A taxa de cruzamento de zero mede quantas vezes o sinal atravessa o eixo zero, sendo um indicador eficaz de percussividade e ruído (Gouyon et al., 2000). Gêneros como rock e metal tendem a apresentar valores mais altos de ZCR.

### 1.3 Características Harmônicas

#### Chroma Features
As características cromáticas capturam a distribuição de energia nas 12 classes de pitch (C, C#, D, etc.), independentemente da oitava (Müller & Ewert, 2011). Estes 12 valores são essenciais para capturar a tonalidade e progressões harmônicas características de cada gênero musical.

#### Tempo (BPM)
O sistema estima o tempo musical em batidas por minuto utilizando o algoritmo de detecção de beats do Librosa. O tempo é uma característica discriminativa importante, pois diferentes gêneros possuem faixas típicas de BPM (Tzanetakis & Cook, 2002).

### 1.4 Normalização de Features

Para garantir que características em diferentes escalas contribuam adequadamente para os cálculos de similaridade, o sistema aplica normalização:
- Centróide e Rolloff espectrais são normalizados pela frequência de Nyquist (sr/2)
- Tempo é normalizado dividindo por 200 BPM
- MFCCs e chroma já são extraídos em escalas apropriadas

## 2. Representação Vetorial e Armazenamento

### 2.1 ChromaDB como Banco de Dados Vetorial

O sistema utiliza ChromaDB (Chroma, 2023), um banco de dados vetorial especializado em embeddings, para armazenamento persistente das características extraídas. Cada música é representada como um vetor de 29 dimensões:
- 13 MFCCs
- 1 Centróide Espectral
- 1 Spectral Rolloff
- 1 Zero Crossing Rate
- 12 Chroma Features
- 1 Tempo

A escolha de um banco de dados vetorial é fundamentada na eficiência de busca por similaridade em espaços de alta dimensionalidade (Johnson et al., 2019). ChromaDB utiliza indexação otimizada (HNSW - Hierarchical Navigable Small World) para realizar buscas aproximadas de vizinhos mais próximos em tempo sublinear (Malkov & Yashunin, 2018).

### 2.2 Persistência e Cache

O sistema implementa persistência local através do ChromaDB, armazenando os vetores de características juntamente com metadados (nome do arquivo e gênero). Um cache em memória é mantido para operações de cálculo manual, permitindo análise detalhada das distâncias.

## 3. Métricas de Similaridade

### 3.1 Distância Euclidiana Ponderada

A distância euclidiana mede a distância geométrica entre dois pontos no espaço vetorial (Deza & Deza, 2009). O sistema implementa uma versão ponderada da distância euclidiana:

```
d(x,y) = √(Σ wᵢ(xᵢ - yᵢ)²)
```

Onde wᵢ representa o peso atribuído a cada característica. Os pesos foram ajustados empiricamente:
- MFCCs: peso 8.0 (críticos para timbre)
- Centróide Espectral: peso 4.0 (brilho)
- Spectral Rolloff: peso 3.5 (conteúdo de frequência)
- Zero Crossing Rate: peso 5.0 (percussão)
- Chroma: peso 2.0 (harmonia)
- Tempo: peso 3.5 (ritmo)

Esta ponderação permite enfatizar características mais discriminativas para classificação de gêneros musicais (Aucouturier & Pachet, 2003).

### 3.2 Similaridade de Cosseno

A similaridade de cosseno mede o ângulo entre dois vetores, sendo invariante à magnitude (Singhal, 2001):

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Onde A · B é o produto escalar dos vetores e ||A|| representa a magnitude (norma L2) do vetor A. Valores próximos a 1 indicam vetores altamente similares. Esta métrica é particularmente útil quando a escala absoluta dos valores é menos relevante que suas proporções relativas (Rahutomo et al., 2012).

## 4. Algoritmo de Classificação

### 4.1 K-Nearest Neighbors (K-NN)

O sistema utiliza o algoritmo K-NN para classificação (Cover & Hart, 1967), um método não-paramétrico baseado em instâncias. O processo de classificação segue as etapas:

1. **Extração de características** da música de teste
2. **Cálculo de distâncias** entre o vetor de teste e todos os vetores no banco
3. **Seleção dos k vizinhos** mais próximos (menor distância)
4. **Votação híbrida** para determinar o gênero

### 4.2 Votação Híbrida

O sistema implementa um mecanismo de votação híbrida que combina:

1. **Votação por contagem**: Cada vizinho contribui com um voto para seu gênero
2. **Votação ponderada por distância**: Vizinhos mais próximos têm maior influência através do peso w = 1/(d + 1)

O score final combina ambas abordagens:
```
Score(g) = 0.5 × (peso_normalizado(g)) + 0.5 × (votos_normalizados(g))
```

Esta abordagem híbrida reduz o impacto de outliers e aumenta a confiabilidade da classificação (Dudani, 1976), sendo mais robusta que votação simples por maioria.

### 4.3 Confiança da Classificação

O sistema calcula um índice de confiança baseado na proporção do score do gênero vencedor em relação à soma total dos scores:

```
Confiança = (Score_vencedor / Σ Scores) × 100%
```

Este métrica fornece feedback sobre a certeza da classificação, sendo útil para identificar casos ambíguos.

## 5. Vantagens da Abordagem

A combinação de ChromaDB com cálculos de similaridade ponderados oferece várias vantagens:

1. **Escalabilidade**: ChromaDB permite buscas eficientes mesmo com grandes volumes de dados
2. **Persistência**: Vetores são armazenados em disco, eliminando necessidade de reprocessamento
3. **Interpretabilidade**: Cálculos manuais permitem análise detalhada das distâncias
4. **Flexibilidade**: Pesos podem ser ajustados para diferentes domínios musicais
5. **Robustez**: Votação híbrida reduz impacto de ruído e outliers

## Referências

Aucouturier, J. J., & Pachet, F. (2003). Representing musical genre: A state of the art. Journal of New Music Research, 32(1), 83-93.

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.

Deza, M. M., & Deza, E. (2009). Encyclopedia of distances. Springer.

Dudani, S. A. (1976). The distance-weighted k-nearest-neighbor rule. IEEE Transactions on Systems, Man, and Cybernetics, 6(4), 325-327.

Gouyon, F., Pachet, F., & Delerue, O. (2000). On the use of zero-crossing rate for an application of classification of percussive sounds. Proceedings of the COST G-6 Conference on Digital Audio Effects (DAFX-00), Verona, Italy.

Grey, J. M. (1977). Multidimensional perceptual scaling of musical timbres. The Journal of the Acoustical Society of America, 61(5), 1270-1277.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, 7(3), 535-547.

Logan, B. (2000). Mel frequency cepstral coefficients for music modeling. In ISMIR (Vol. 270, pp. 1-11).

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824-836.

McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. In Proceedings of the 14th Python in Science Conference (Vol. 8, pp. 18-25).

Müller, M., & Ewert, S. (2011). Chroma toolbox: MATLAB implementations for extracting variants of chroma-based audio features. In Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR).

Rahutomo, F., Kitasuka, T., & Aritsugi, M. (2012). Semantic cosine similarity. In The 7th International Student Conference on Advanced Science and Technology (ICAST) (Vol. 4, No. 1, p. 1).

Scheirer, E. D. (1998). Tempo and beat analysis of acoustic musical signals. The Journal of the Acoustical Society of America, 103(1), 588-601.

Singhal, A. (2001). Modern information retrieval: A brief overview. IEEE Data Engineering Bulletin, 24(4), 35-43.

Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
