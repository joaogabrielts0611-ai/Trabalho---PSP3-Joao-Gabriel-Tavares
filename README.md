# Preços do dia a dia nos EUA (BLS, 2015–2026): auditoria, inflação real e correlações entre pares

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaogabrielts0611-ai/Trabalho---PSP3-Joao-Gabriel-Tavares/blob/main/analise_precos_eua.ipynb)
[![Ver no nbviewer](https://img.shields.io/badge/ver%20com%20gr%C3%A1ficos-nbviewer-orange)](https://nbviewer.org/github/joaogabrielts0611-ai/Trabalho---PSP3-Joao-Gabriel-Tavares/blob/main/analise_precos_eua.ipynb)

**Autor:** João Gabriel Tavares Silva

Análise exploratória e estatística da base *US Grocery and Gas Prices (2015-2026)* (Kaggle), que reúne 74 itens do programa **Average Price Data** do US Bureau of Labor Statistics: ovos, leite, carnes, café, gasolina, diesel, eletricidade etc., mensais de jan/2015 a jul/2026, em dólares.

## Como abrir

| Onde | Para quê |
|---|---|
| **Colab** (botão acima) | Rodar tudo e usar o painel interativo: `Ambiente de execução → Executar tudo`. Nada precisa ser instalado; os dados são baixados deste repositório. |
| **nbviewer** (botão acima) | Ler o notebook já executado, com todos os gráficos interativos, sem rodar nada. |
| `relatorio_analise.html` | O notebook executado em uma página HTML única: baixe e abra no navegador (precisa de internet para carregar a biblioteca de gráficos). |
| GitHub | O GitHub mostra o texto e as tabelas, mas **não** desenha os gráficos Plotly. Use um dos links acima. |

## O que o notebook faz

1. **Auditoria da base.** Confere o README do Kaggle e encontra seis problemas: café classificado como carne bovina; "médias dos primeiros 12 meses" que na verdade cobrem até 113 meses de calendário; bases de 2015 e 2018 misturadas numa mesma mediana; séries vivas marcadas como descontinuadas; inflação do período subestimada (38,5%, não 35%); e período-base dos ovos contaminado pelo surto de gripe aviária de 2015.
2. **Análise inicial aprimorada.** Variação nominal e **real** (deflacionada pelo CPI-U), janelas de calendário comparáveis, taxa anual, volatilidade × variação real, sazonalidade, regimes (pré-pandemia, choque, pós-choque), o caso dos ovos e a ordem de repasse do choque de 2022.
3. **Correlações entre pares.** Todos os ~1.900 pares de itens ativos, em três medidas (nível, variação em 12 meses, variação mensal dessazonalizada), com teste de significância corrigido por Benjamini-Hochberg, robustez (Spearman e exclusão de 2020), matriz clusterizada, rede de co-movimento, PCA e correlações em janela móvel.
4. **Defasagens.** Energia antecipa alimentos? Correlação cruzada, teste de Granger e regressão com erros HAC.
5. **Painel interativo.** Explorador de item, comparador de qualquer par, ranking filtrável de pares e cesta personalizada.

## Principais achados

* Todos os 62 itens com dados suficientes subiram em dólares correntes (mediana +42%), mas só **35 de 62** subiram acima da inflação. Mediana real: +3,8%.
* Café (+43% real), refrigerante em lata e suco de laranja lideram; macarrão, limão e costeleta suína foram os que mais ficaram baratos em termos reais.
* **A maior parte da correlação entre preços é tendência comum.** Em nível, 86% dos pares têm |r| > 0,5; em variação mensal dessazonalizada, só 3%.
* As relações robustas são dentro da mesma cadeia (cortes bovinos, combustíveis). Carne suína × bovina é o par mais forte entre grupos, mas parte dele vem do fechamento de frigoríficos em 2020.
* A correlação entre alimentos e energia **mudou de sinal** depois de 2022. A inflação do diesel tem correlação máxima com a de alimentos 5 meses depois, mas a precedência estatística (Granger) é fraca.

## Estrutura

```
analise_precos_eua.ipynb   notebook principal (executado)
analise_precos_eua.py      mesmo notebook em formato texto (jupytext), para revisar diferenças
relatorio_analise.html     versão HTML do notebook executado
data/
  us_average_prices_monthly.csv        formato longo (item × mês)
  us_average_prices_wide.csv           formato largo (mês × item)
  us_average_prices_item_summary.csv   resumo por item (do Kaggle)
  cpi_u_nsa_CPIAUCNS.csv               CPI-U, sem ajuste sazonal (FRED CPIAUCNS)
  README_kaggle_original.md            descrição original da base
requirements.txt
```

## Rodar localmente

```bash
pip install -r requirements.txt
jupyter lab analise_precos_eua.ipynb
```

## Fontes e licença

* U.S. Bureau of Labor Statistics, Average Price Data (AP), U.S. city average, mensal, sem ajuste sazonal. Obras do governo federal dos EUA são de domínio público.
* CPI-U, All items, U.S. city average, NSA (série FRED CPIAUCNS). Out/2025 não foi publicado (paralisação do governo) e é interpolado no notebook.
* O BLS não endossa esta compilação nem esta análise.

## Short paper (LaTeX / Overleaf)

O short paper em ABNT (classe abnTeX2) está na pasta `shortpaper/`: `main.tex`, `referencias.bib`, figuras e o PDF compilado (`shortpaper.pdf`).

[![Abrir no Overleaf](https://img.shields.io/badge/abrir%20no-Overleaf-47A141)](https://www.overleaf.com/docs?snip_uri=https://raw.githubusercontent.com/joaogabrielts0611-ai/Trabalho---PSP3-Joao-Gabriel-Tavares/main/shortpaper-overleaf.zip)

O botão acima cria uma cópia do projeto na conta Overleaf de quem clicar. Compilador: pdfLaTeX (padrão).
