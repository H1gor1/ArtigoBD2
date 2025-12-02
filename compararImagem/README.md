# Sistema de Comparação de Imagens

Sistema para comparação e busca de imagens similares usando vetorização de features e bancos de dados vetoriais (ChromaDB ou FAISS).

## 📋 Requisitos

- Python 3.8+
- pip

## 🚀 Instalação

1. Clone ou baixe o projeto

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 📖 Como Usar

O sistema possui 4 comandos principais:

### 1. Demo - Demonstração Matemática

Executa uma demonstração com vetores simples para entender o cálculo de similaridade cosseno:

```bash
python main.py demo
```

### 2. Indexar - Adicionar Imagens ao Banco

**Indexar uma única imagem:**
```bash
python main.py indexar --imagem caminho/para/imagem.jpg
```

**Indexar todas as imagens de uma pasta:**
```bash
python main.py indexar --pasta fotos/
```

**Escolher o banco de dados (ChromaDB ou FAISS):**
```bash
python main.py --banco faiss indexar --pasta fotos/
python main.py --banco chroma indexar --pasta fotos/
```

### 3. Buscar - Encontrar Imagens Similares

**Busca básica (3 resultados mais similares):**
```bash
python main.py buscar fotos/cachorro1.jpeg
```

**Buscar com mais resultados:**
```bash
python main.py buscar fotos/cachorro1.jpeg --top 5
```

**Buscar e visualizar gráfico:**
```bash
python main.py buscar fotos/cachorro1.jpeg --plot
```

**Buscar e salvar gráfico:**
```bash
python main.py buscar fotos/cachorro1.jpeg --salvar resultado_busca.png
```

### 4. Comparar - Comparar Duas Imagens

**Comparação básica:**
```bash
python main.py comparar fotos/cachorro1.jpeg fotos/cachorro2.jpg
```

**Comparação com detalhes matemáticos:**
```bash
python main.py comparar fotos/cachorro1.jpeg fotos/cachorro2.jpg --detalhes
```

**Comparação com visualização:**
```bash
python main.py comparar fotos/gatoCinza.jpg fotos/gatoPreto.jpg --plot
```

**Comparação e salvar gráfico:**
```bash
python main.py comparar fotos/gatoCinza.jpg fotos/gatoPreto.jpg --salvar comparacao.png
```

## 🔧 Estrutura do Projeto

```
compararImagem/
├── main.py                 # Arquivo principal
├── requirements.txt        # Dependências
├── fotos/                 # Pasta com imagens de exemplo
│   ├── cachorro1.jpeg
│   ├── cachorro2.jpg
│   ├── gatoCinza.jpg
│   ├── gatoLaranja.jpg
│   └── gatoPreto.jpg
└── src/
    ├── comparador.py      # Lógica principal de comparação
    ├── models/            # Extração de features
    │   └── extrator.py
    ├── database/          # Adaptadores de banco de dados
    │   ├── chromadb_adapter.py
    │   └── faiss_adapter.py
    └── utils/             # Utilitários
        ├── matematica.py       # Cálculos matemáticos
        └── visualizacao.py     # Geração de gráficos
```

## 💡 Exemplos de Uso

### Exemplo Completo: Indexar e Buscar

```bash
# 1. Indexar todas as imagens
python main.py indexar --pasta fotos/

# 2. Buscar imagens similares a um cachorro
python main.py buscar fotos/cachorro1.jpeg --top 3 --plot

# 3. Comparar dois gatos
python main.py comparar fotos/gatoCinza.jpg fotos/gatoPreto.jpg --detalhes --plot
```

### Exemplo com FAISS

```bash
# Usar FAISS ao invés de ChromaDB
python main.py --banco faiss indexar --pasta fotos/
python main.py --banco faiss buscar fotos/gatoLaranja.jpg --top 3
```

## 📊 Bancos de Dados Suportados

- **ChromaDB** (padrão): Banco de dados vetorial com persistência automática
- **FAISS**: Biblioteca de busca de similaridade do Facebook, otimizada para alta performance

## 🎨 Visualizações

O sistema pode gerar gráficos mostrando:
- Comparação visual entre imagens
- Scores de similaridade
- Resultados de busca ranqueados
- Análise detalhada de vetores de features

## ⚙️ Parâmetros Disponíveis

### Opções Globais
- `--banco {chroma,faiss}`: Escolhe o banco de dados vetorial (padrão: chroma)

### Comando `indexar`
- `--imagem`: Caminho para uma única imagem
- `--pasta`: Caminho para uma pasta com imagens

### Comando `buscar`
- `imagem`: Caminho da imagem de consulta (obrigatório)
- `--top`: Quantidade de resultados (padrão: 3)
- `--plot`: Mostrar gráficos
- `--salvar`: Salvar gráfico em arquivo

### Comando `comparar`
- `imagem1`: Primeira imagem (obrigatório)
- `imagem2`: Segunda imagem (obrigatório)
- `--detalhes`: Mostrar cálculo passo a passo
- `--plot`: Mostrar gráficos
- `--salvar`: Salvar gráfico em arquivo

## 📝 Notas

- As imagens indexadas são armazenadas persistentemente no banco escolhido
- O sistema usa ResNet50 pré-treinado para extrair features das imagens
- A similaridade é calculada usando cosseno entre vetores de features
- Formatos suportados: JPG, JPEG, PNG, BMP, GIF
