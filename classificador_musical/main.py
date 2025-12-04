"""Interface CLI do classificador."""

from classificador import ClassificadorMusical
from downloader import Downloader


def main():
    clf = ClassificadorMusical()
    downloader = Downloader()

    while True:
        print("\n" + "=" * 50)
        print("CLASSIFICADOR DE GÊNEROS MUSICAIS")
        print("=" * 50)
        print("1. Adicionar música")
        print("2. Classificar música (com cálculos)")
        print("3. Baixar música para classificar")
        print("4. Sair")
        print("=" * 50)

        opcao = input("\nOpção: ").strip()

        if opcao == "1":
            print("\n--- ADICIONAR MÚSICA ---")
            caminho = input("Caminho do arquivo: ").strip()
            genero = input("Gênero (ex: rock, jazz, pop): ").strip()

            if caminho and genero:
                try:
                    clf.adicionar_musica(caminho, genero)
                except Exception as e:
                    print(f"Erro: {e}")
            else:
                print("Caminho e gênero são obrigatórios!")

        elif opcao == "2":
            print("\n--- CLASSIFICAR MÚSICA ---")
            caminho = input("Caminho do arquivo: ").strip()
            k = input("Número de vizinhos (padrão 5): ").strip()
            k = int(k) if k else 5

            if caminho:
                try:
                    clf.classificar_musica(caminho, k=k, mostrar_calculos=True)
                except Exception as e:
                    print(f"Erro: {e}")
            else:
                print("Caminho é obrigatório!")
        elif opcao == "3":
            print("Baixando música...")
            link = input("Link da música: ").strip()
            if link:
                try:
                    downloader.download(link)
                except Exception as e:
                    print(f"Erro: {e}")
            else:
                print("Link é obrigatório!")
        elif opcao == "4":
            print("\nAté logo!")
            break

        else:
            print("Opção inválida!")


if __name__ == "__main__":
    main()
