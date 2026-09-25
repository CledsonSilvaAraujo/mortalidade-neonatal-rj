"""Gera README.md e slides/arguicao.md a partir dos templates, com as mesmas macros do relatório.

Cada `{{nome}}` do template é trocado pelo valor de `relatorio/macros.json` (escrito por
gerar_numeros.py), convertido de LaTeX para Markdown. Placeholder sem macro é erro.
Uso: .venv/bin/python relatorio/gerar_readme.py
"""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
MACROS = json.loads((RAIZ / "relatorio" / "macros.json").read_text(encoding="utf-8"))
# Macros da v2 (aprofundamento), escritas por relatorio_v2/gerar_numeros_v2.py.
_V2 = RAIZ / "relatorio_v2" / "macros_v2.json"
if _V2.exists():
    MACROS.update(json.loads(_V2.read_text(encoding="utf-8")))


def markdown(v: str) -> str:
    v = re.sub(r"\\times 10\^\{-(\d+)\}", r"×10⁻\1", v)
    v = v.replace("\\%", "%").replace("~", " ").replace("--", "–").replace("$-$", "−")
    return v.replace("-0,", "−0,").replace("(-", "(−").replace(" -", " −")


PARES = [("README.tmpl.md", "README.md"), ("slides/arguicao.tmpl.md", "slides/arguicao.md")]


def main() -> None:
    for origem, destino in PARES:
        tmpl = (RAIZ / origem).read_text(encoding="utf-8")
        faltam = sorted({m for m in re.findall(r"\{\{(\w+)\}\}", tmpl) if m not in MACROS})
        if faltam:
            raise SystemExit(f"{origem}: macros inexistentes: {faltam}")
        saida = re.sub(r"\{\{(\w+)\}\}", lambda m: markdown(MACROS[m.group(1)]), tmpl)
        (RAIZ / destino).write_text(saida, encoding="utf-8")
        print(f"{destino} gerado ({len(re.findall(r'{{', tmpl))} valores)")


if __name__ == "__main__":
    main()
