"""Script para popular o banco vetorial com músicas de treino."""
import os
from classificador import ClassificadorMusical

def popular_banco():
    """Adiciona todas as músicas de treino ao banco."""
    classificador = ClassificadorMusical()
    base_path = "./musicas_teste"
    
    generos = {
        'rock': 'rock',
        'pop': 'pop',
        'jazz': 'jazz',
        'eletronica': 'eletronica',
        'classica': 'classica'
    }
    
    print("="*50)
    print("POPULANDO BANCO VETORIAL")
    print("="*50)
    
    total = 0
    for pasta, genero in generos.items():
        caminho_pasta = os.path.join(base_path, pasta)
        if not os.path.exists(caminho_pasta):
            print(f"Pasta {caminho_pasta} não encontrada!")
            continue
        
        print(f"\n--- Gênero: {genero.upper()} ---")
        arquivos = [f for f in os.listdir(caminho_pasta) if f.endswith('.mp3')]
        
        for arquivo in sorted(arquivos):
            caminho = os.path.join(caminho_pasta, arquivo)
            try:
                classificador.adicionar_musica(caminho, genero)
                total += 1
            except Exception as e:
                print(f"Erro ao adicionar {arquivo}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Total de músicas adicionadas: {total}")
    print(f"{'='*50}")

if __name__ == "__main__":
    popular_banco()
