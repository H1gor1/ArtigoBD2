# Metodologia: Sistema de Recuperação de Imagens por Similaridade

## 1. Introdução

O sistema desenvolvido implementa um mecanismo de recuperação de imagens baseado em conteúdo (Content-Based Image Retrieval - CBIR), utilizando técnicas de aprendizado profundo para extração de características visuais e bancos de dados vetoriais para indexação e busca eficiente. Esta abordagem se fundamenta em três pilares principais: extração de features através de redes neurais convolucionais, cálculo de similaridade através de métricas matemáticas, e armazenamento otimizado em estruturas de dados especializadas.

## 2. Extração de Características (Feature Extraction)

### 2.1 Arquitetura ResNet50

Para a extração de características visuais, o sistema utiliza a arquitetura ResNet50 (He et al., 2016), uma rede neural convolucional profunda pré-treinada no dataset ImageNet. A ResNet50 é composta por 50 camadas e emprega conexões residuais (skip connections) que permitem o treinamento de redes mais profundas sem o problema de degradação do gradiente (He et al., 2016).

No contexto deste trabalho, a camada de classificação final da rede foi removida, utilizando-se apenas as camadas convolucionais como extrator de features. Esta técnica, conhecida como transfer learning, permite obter representações vetoriais de 2048 dimensões que codificam características visuais de alto nível das imagens (Razavian et al., 2014).

### 2.2 Pipeline de Pré-processamento

Antes da extração de features, cada imagem passa por um pipeline de transformações padronizado:

1. **Redimensionamento**: A imagem é redimensionada para 256×256 pixels
2. **Crop Central**: Extração de uma região central de 224×224 pixels, dimensão padrão da ResNet50
3. **Normalização**: Aplicação de normalização z-score com média μ = [0.485, 0.456, 0.406] e desvio padrão σ = [0.229, 0.224, 0.225] para os canais RGB, valores derivados do dataset ImageNet (Deng et al., 2009)

### 2.3 Normalização de Features

Após a extração, os vetores de características são normalizados para norma unitária (L2 = 1), seguindo a equação:

```
v̂ = v / ||v||₂
```

onde v é o vetor original e ||v||₂ representa a norma euclidiana. Esta normalização é crucial para garantir que a similaridade por cosseno seja equivalente ao produto interno, simplificando os cálculos posteriores (Wan et al., 2014).

## 3. Métricas de Similaridade

### 3.1 Similaridade de Cosseno

A similaridade entre duas imagens é quantificada através da similaridade de cosseno, uma métrica amplamente utilizada em sistemas de recuperação de informação (Salton & McGill, 1983). Dados dois vetores de características v₁ e v₂, a similaridade de cosseno é definida como:

```
cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)
```

onde:
- v₁ · v₂ representa o produto escalar (dot product) dos vetores
- ||v₁|| e ||v₂|| são as magnitudes (normas) dos vetores
- θ é o ângulo entre os vetores no espaço multidimensional

Esta métrica produz valores no intervalo [-1, 1], onde 1 indica máxima similaridade (vetores idênticos), 0 indica ortogonalidade (sem correlação), e -1 indica dissimilaridade máxima (Singhal, 2001).

### 3.2 Vantagens da Similaridade de Cosseno

A escolha da similaridade de cosseno sobre outras métricas, como a distância euclidiana, apresenta vantagens específicas para o domínio de imagens:

1. **Invariância à Magnitude**: O cosseno é independente da magnitude dos vetores, focando apenas na direção. Isso é particularmente útil quando as imagens podem ter diferenças de brilho ou contraste (Huang, 2008).

2. **Eficiência Computacional**: Com vetores normalizados, o cálculo se reduz a um simples produto interno, operação altamente otimizada em hardware moderno (Manning et al., 2008).

3. **Interpretabilidade**: Os valores resultantes têm interpretação geométrica clara, representando o cosseno do ângulo entre vetores no espaço de características.

### 3.3 Relação com Distância Euclidiana

Embora não seja a métrica principal utilizada, a distância euclidiana pode ser derivada da similaridade de cosseno para vetores normalizados através da relação:

```
d_euclidiana = √(2 - 2 × cos(θ))
```

Esta conversão é útil para alguns bancos de dados vetoriais que trabalham nativamente com métricas de distância (Johnson et al., 2019).

## 4. Bancos de Dados Vetoriais

O sistema oferece suporte a duas tecnologias distintas de armazenamento e recuperação vetorial, cada uma com características específicas.

### 4.1 ChromaDB

ChromaDB é um banco de dados vetorial open-source otimizado para embeddings de machine learning (Trychta, 2023). No contexto deste trabalho, utiliza-se o ChromaDB com as seguintes configurações:

- **Algoritmo de Indexação**: HNSW (Hierarchical Navigable Small World)
- **Métrica de Distância**: Cosseno
- **Persistência**: Modo persistente com armazenamento em disco

O algoritmo HNSW é baseado em grafos hierárquicos navegáveis, proporcionando busca aproximada com complexidade logarítmica O(log n) e alta recall, sendo especialmente eficiente para datasets de média a grande escala (Malkov & Yashunin, 2018).

**Vantagens do ChromaDB**:
- Interface Python nativa e intuitiva
- Persistência automática de dados
- Suporte nativo a metadados
- Atualizações e remoções eficientes

### 4.2 FAISS (Facebook AI Similarity Search)

FAISS é uma biblioteca desenvolvida pelo Facebook AI Research para busca eficiente de similaridade em vetores densos de alta dimensionalidade (Johnson et al., 2019). A implementação utiliza:

- **Tipo de Índice**: IndexFlatIP (Flat Index with Inner Product)
- **Precisão**: Busca exata (não aproximada)
- **Tipo de Dados**: Float32 para otimização de memória

O IndexFlatIP realiza busca exaustiva calculando o produto interno entre o vetor de consulta e todos os vetores indexados. Embora tenha complexidade O(n), é garantidamente exato e serve como baseline para comparação com métodos aproximados (Johnson et al., 2019).

**Vantagens do FAISS**:
- Performance otimizada com SIMD e GPU
- Suporte a bilhões de vetores
- Múltiplos algoritmos de indexação disponíveis
- Amplamente validado em produção (usado no Facebook, Google)

### 4.3 Comparação entre ChromaDB e FAISS

| Característica | ChromaDB | FAISS |
|----------------|----------|-------|
| Algoritmo | HNSW (aproximado) | Flat Index (exato) |
| Complexidade de Busca | O(log n) | O(n) |
| Recall | ~99% (configurável) | 100% |
| Facilidade de Uso | Alta | Média |
| Escalabilidade | Boa (até milhões) | Excelente (bilhões) |
| Persistência | Nativa | Manual |

## 5. Workflow do Sistema

O sistema opera através de três operações principais:

### 5.1 Indexação

Durante a fase de indexação:

1. A imagem é carregada e pré-processada
2. O vetor de features (2048-D) é extraído via ResNet50
3. O vetor é normalizado para norma unitária
4. O embedding é inserido no banco de dados vetorial com metadados (path, nome)

### 5.2 Busca por Similaridade

Para recuperação de imagens similares:

1. A imagem de consulta é processada e vetorizada
2. O banco de dados executa busca de vizinhos mais próximos (k-NN)
3. Os top-k resultados mais similares são retornados
4. Scores de similaridade são calculados e ordenados

### 5.3 Comparação Direta

Para comparação entre duas imagens específicas:

1. Ambas as imagens são vetorizadas
2. A similaridade de cosseno é calculada explicitamente
3. Métricas detalhadas podem ser exibidas (produto escalar, normas, etc.)

## 6. Fundamentação Teórica

### 6.1 Aprendizado Profundo para Extração de Features

O uso de redes neurais convolucionais profundas para extração de características visuais é fundamentado em décadas de pesquisa em visão computacional. Krizhevsky et al. (2012) demonstraram que CNNs profundas podem aprender hierarquias de features, desde bordas simples nas camadas iniciais até conceitos semânticos complexos nas camadas finais.

A técnica de transfer learning, empregada neste trabalho, foi validada por Yosinski et al. (2014), que demonstraram que features aprendidas em tarefas de classificação generalizam bem para outras tarefas visuais, incluindo busca por similaridade.

### 6.2 Busca Aproximada de Vizinhos Mais Próximos

O problema de busca de vizinhos mais próximos (k-NN search) em espaços de alta dimensionalidade é intrinsecamente complexo devido à "maldição da dimensionalidade" (Indyk & Motwani, 1998). Algoritmos como HNSW oferecem trade-offs favoráveis entre precisão e performance através de aproximações controladas (Malkov & Yashunin, 2018).

### 6.3 Espaços de Embedding

A representação de imagens em espaços vetoriais contínuos permite a aplicação de operações algébricas para capturar relações semânticas. Wan et al. (2014) demonstraram que embeddings aprendidos por CNNs formam espaços métricos onde a distância correlaciona fortemente com similaridade visual perceptual.

## 7. Considerações de Implementação

### 7.1 Eficiência Computacional

O sistema foi projetado considerando eficiência:
- Processamento em lote de imagens durante indexação
- Uso de PyTorch com otimizações CUDA quando disponível
- Caching de modelos pré-treinados
- Normalização prévia dos vetores para simplificar cálculos

### 7.2 Escalabilidade

A arquitetura modular permite diferentes estratégias de escalabilidade:
- ChromaDB para aplicações com até milhões de imagens
- FAISS com índices aproximados (IVF, PQ) para bilhões de vetores
- Possibilidade de sharding e distribuição em múltiplas máquinas

### 7.3 Qualidade dos Resultados

A qualidade da recuperação depende de vários fatores:
- Domínio das imagens (ImageNet bias)
- Qualidade das imagens de entrada
- Tamanho do dataset indexado
- Parâmetros de configuração dos algoritmos de busca

## 8. Conclusão

O sistema implementado representa uma solução robusta e escalável para recuperação de imagens por similaridade, combinando técnicas estado-da-arte de aprendizado profundo com bancos de dados vetoriais modernos. A escolha da ResNet50 para extração de features e da similaridade de cosseno como métrica de comparação são bem fundamentadas na literatura acadêmica, enquanto o suporte a múltiplos backends (ChromaDB e FAISS) oferece flexibilidade para diferentes cenários de uso.

## Referências

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248-255.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

Huang, A. (2008). Similarity measures for text document clustering. *Proceedings of the Sixth New Zealand Computer Science Research Student Conference (NZCSRSC2008)*, 49-56.

Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality. *Proceedings of the Thirtieth Annual ACM Symposium on Theory of Computing*, 604-613.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105.

Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press.

Razavian, A. S., Azizpour, H., Sullivan, J., & Carlsson, S. (2014). CNN features off-the-shelf: an astounding baseline for recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, 806-813.

Salton, G., & McGill, M. J. (1983). *Introduction to modern information retrieval*. McGraw-Hill.

Singhal, A. (2001). Modern information retrieval: A brief overview. *IEEE Data Engineering Bulletin*, 24(4), 35-43.

Trychta, A. (2023). ChromaDB: The AI-native open-source embedding database. *GitHub Repository*. https://github.com/chroma-core/chroma

Wan, J., Wang, D., Hoi, S. C. H., Wu, P., Zhu, J., Zhang, Y., & Li, J. (2014). Deep learning for content-based image retrieval: A comprehensive study. *Proceedings of the 22nd ACM International Conference on Multimedia*, 157-166.

Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *Advances in Neural Information Processing Systems*, 27, 3320-3328.
