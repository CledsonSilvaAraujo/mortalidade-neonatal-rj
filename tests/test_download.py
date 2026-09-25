from src import download


def test_url_usa_pasta_preliminar_so_quando_listada():
    assert "/SIM/PRELIM/DORES/DORJ2025.dbc" in download.url_arquivo("DO", 2025)
    assert "/SIM/CID10/DORES/DORJ2024.dbc" in download.url_arquivo("DO", 2024)
    assert "/SINASC/1996_/Dados/DNRES/DNRJ2024.dbc" in download.url_arquivo("DN", 2024)
