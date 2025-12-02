import argparse
import numpy as np
from src.comparador import ComparadorDeImagens
from src.utils import SimilarityCalculator


def main():
    parser = argparse.ArgumentParser(description='Sistema de Comparação de Imagens')
    parser.add_argument('--banco', choices=['chroma', 'faiss'], default='chroma',
                        help='Escolha o banco de dados vetorial (padrão: chroma)')
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponíveis')
    
    # Comando: indexar
    parser_indexar = subparsers.add_parser('indexar', help='Indexar imagens no banco')
    parser_indexar.add_argument('--imagem', type=str, help='Caminho para uma imagem')
    parser_indexar.add_argument('--pasta', type=str, help='Caminho para uma pasta com imagens')
    
    # Comando: buscar
    parser_buscar = subparsers.add_parser('buscar', help='Buscar imagens similares')
    parser_buscar.add_argument('imagem', type=str, help='Caminho da imagem de consulta')
    parser_buscar.add_argument('--top', type=int, default=3, help='Quantidade de resultados (padrão: 3)')
    parser_buscar.add_argument('--plot', action='store_true', help='Mostrar gráficos dos resultados')
    parser_buscar.add_argument('--salvar', type=str, help='Salvar gráfico em arquivo (ex: busca.png)')
    
    # Comando: comparar
    parser_comparar = subparsers.add_parser('comparar', help='Comparar duas imagens')
    parser_comparar.add_argument('imagem1', type=str, help='Primeira imagem')
    parser_comparar.add_argument('imagem2', type=str, help='Segunda imagem')
    parser_comparar.add_argument('--detalhes', action='store_true', help='Mostrar cálculo passo a passo')
    parser_comparar.add_argument('--plot', action='store_true', help='Mostrar gráficos de análise')
    parser_comparar.add_argument('--salvar', type=str, help='Salvar gráfico em arquivo (ex: resultado.png)')
    
    # Comando: demo
    parser_demo = subparsers.add_parser('demo', help='Executar demonstração com vetores simples')
    
    args = parser.parse_args()
    
    if args.comando is None:
        parser.print_help()
        return
    
    # Demonstração matemática
    if args.comando == 'demo':
        print("--- TESTE DE CÁLCULO MANUAL (Exemplo Simples) ---")
        v_teste_a = np.array([1, 0, 1])
        v_teste_b = np.array([0, 1, 1])
        v_teste_a = v_teste_a / np.linalg.norm(v_teste_a)
        v_teste_b = v_teste_b / np.linalg.norm(v_teste_b)
        SimilarityCalculator.calcular_similaridade_cosseno(v_teste_a, v_teste_b)
        return
    
    # Inicializar sistema
    sistema = ComparadorDeImagens(usar_banco=args.banco)
    
    # Executar comandos
    if args.comando == 'indexar':
        if args.imagem:
            sistema.indexar_imagem(args.imagem)
        elif args.pasta:
            sistema.indexar_pasta(args.pasta)
        else:
            print("Erro: forneça --imagem ou --pasta")
    
    elif args.comando == 'buscar':
        resultados = sistema.buscar_similares(
            args.imagem, 
            top_k=args.top,
            visualizar=args.plot,
            salvar_plot=args.salvar
        )
        print("\n--- RESULTADOS ---")
        for i, res in enumerate(resultados, 1):
            print(f"{i}. Imagem: {res['id']} | Similaridade: {res['similaridade']:.4f}")
    
    elif args.comando == 'comparar':
        sistema.comparar_duas_imagens(
            args.imagem1, 
            args.imagem2, 
            mostrar_detalhes=args.detalhes,
            visualizar=args.plot,
            salvar_plot=args.salvar
        )


if __name__ == "__main__":
    main()
