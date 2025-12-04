# Classificador de Gêneros Musicais

Sistema simples para classificar músicas por gênero usando ChromaDB.

## Instalação

```bash
pip install -r requirements.txt
```

## Como Usar

### Interface de Linha de Comando

```bash
python main.py
```

**Opções:**
1. Adicionar música - Adiciona uma música ao banco com seu gênero
2. Classificar música - Descobre o gênero de uma música nova
3. Sair

### Uso no Código

```python
from classificador import ClassificadorMusical

clf = ClassificadorMusical()

# Adiciona músicas
clf.adicionar_musica('rock1.mp3', 'rock')
clf.adicionar_musica('jazz1.mp3', 'jazz')

# Classifica uma música nova
genero = clf.classificar_musica('teste.mp3', k=5)
print(f"Gênero: {genero}")
```

## Como Funciona

1. **Extração de Features**: Extrai 29 características da música (MFCC, Centroid, Rolloff, etc)
2. **Armazenamento**: Salva os vetores no ChromaDB (banco de dados vetorial)
3. **Classificação**: Calcula a distância euclidiana simples para encontrar as k músicas mais similares e vota no gênero

## Estrutura

- `extrator_features.py` - Extrai features das músicas
- `banco_vetorial.py` - Interface com ChromaDB
- `classificador.py` - Lógica de classificação
- `main.py` - Interface de linha de comando
