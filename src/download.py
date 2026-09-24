"""Download e leitura dos microdados públicos do SINASC e do SIM (DATASUS, formato DBC)."""
from pathlib import Path
import urllib.request

import pandas as pd
import pyreaddbc
from dbfread import DBF

# Caminhos ancorados na raiz do projeto: funcionam a partir de notebook/ ou da raiz.
DADOS = Path(__file__).resolve().parent.parent / "data"
RAW = DADOS / "raw"

UF = "RJ"
ANOS_NASCIMENTO = range(2017, 2024)   # nascidos vivos 2017–2023
ANOS_OBITO = range(2017, 2025)        # óbitos até 2024: quem nasce em dezembro morre no ano seguinte

FTP = "ftp://ftp.datasus.gov.br/dissemin/publicos"
PASTAS = {"DN": "SINASC/1996_/Dados/DNRES", "DO": "SIM/CID10/DORES"}


def nome_arquivo(sistema: str, ano: int, uf: str = UF) -> str:
    return f"{sistema}{uf}{ano}.dbc"


def url_arquivo(sistema: str, ano: int, uf: str = UF) -> str:
    return f"{FTP}/{PASTAS[sistema]}/{nome_arquivo(sistema, ano, uf)}"


def baixar_arquivo(sistema: str, ano: int, destino: Path,
                   baixador=urllib.request.urlretrieve, tentativas: int = 3) -> Path:
    """Baixa um DBC para `destino`, reaproveitando o cache se já existir.

    O download vai para um `.part` e só é renomeado no fim, para que uma
    interrupção nunca deixe um arquivo incompleto passando por válido. O FTP do
    DATASUS cai com frequência, por isso há novas tentativas.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    final = destino / nome_arquivo(sistema, ano)
    if final.exists():
        return final
    parcial = final.with_name(final.name + ".part")
    for tentativa in range(1, tentativas + 1):
        try:
            baixador(url_arquivo(sistema, ano), parcial)
            break
        except BaseException:
            parcial.unlink(missing_ok=True)
            if tentativa == tentativas:
                raise
    parcial.rename(final)
    return final


def ler_dbc(caminho: Path) -> pd.DataFrame:
    """Lê um DBC como texto (todas as colunas `str`); o cache em parquet evita reconverter."""
    caminho = Path(caminho)
    cache = caminho.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    dbf = caminho.with_suffix(".dbf")
    pyreaddbc.dbc2dbf(str(caminho), str(dbf))
    df = pd.DataFrame(iter(DBF(dbf, encoding="latin-1", char_decode_errors="replace")))
    df = df.astype("string").apply(lambda s: s.str.strip()).replace("", pd.NA)
    df.to_parquet(cache)
    dbf.unlink()
    return df


def baixar_tudo(destino: Path = RAW) -> dict[str, list[Path]]:
    return {
        "DN": [baixar_arquivo("DN", ano, destino) for ano in ANOS_NASCIMENTO],
        "DO": [baixar_arquivo("DO", ano, destino) for ano in ANOS_OBITO],
    }


def carregar(sistema: str, anos, pasta: Path = RAW) -> pd.DataFrame:
    return pd.concat([ler_dbc(Path(pasta) / nome_arquivo(sistema, a)).assign(ARQUIVO_ANO=a)
                      for a in anos], ignore_index=True)


if __name__ == "__main__":
    for sistema, arquivos in baixar_tudo().items():
        for arq in arquivos:
            print(sistema, arq, len(ler_dbc(arq)))
