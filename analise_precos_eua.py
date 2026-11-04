# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quanto custa o dia a dia nos EUA? Preços médios do BLS, 2015–2026
#
# **Autor:** João Gabriel Tavares Silva
#
# [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joaogabrielts0611-ai/analise-precos-eua-bls/blob/main/analise_precos_eua.ipynb)
#
# **Base:** *US Grocery and Gas Prices (2015-2026)* (Kaggle), compilada a partir do programa **Average Price Data (AP)** do US Bureau of Labor Statistics. São 74 itens, mensais, de jan/2015 a jul/2026, em dólares correntes (não é índice), média nacional ("U.S. city average"), sem ajuste sazonal.
#
# **O que este notebook faz**
#
# 1. **Auditoria da base.** Confere as afirmações do README do Kaggle e encontra problemas que mudam os resultados.
# 2. **Análise inicial aprimorada.** Variação nominal *e real* (deflacionada pelo CPI-U), janelas de calendário comparáveis, taxa anual, volatilidade, sazonalidade, regimes e o caso dos ovos.
# 3. **Correlações entre pares.** Todas as combinações de itens ativos (cerca de 1.900 pares), com a armadilha da correlação espúria tratada, teste de significância com correção para múltiplas comparações, clusters, rede de co-movimento e PCA.
# 4. **Defasagens.** O preço do diesel e da gasolina antecipa o preço dos alimentos? (correlação cruzada e teste de Granger)
# 5. **Painel interativo.** Explorador de item, comparador de qualquer par, ranking filtrável de pares e cesta personalizada.
#
# > **Como usar:** no Colab, `Ambiente de execução → Executar tudo`. Os gráficos são interativos (passe o mouse, clique na legenda para esconder séries, arraste para dar zoom). Os controles da Seção 7 precisam do notebook em execução.

# %% [markdown]
# ## 0. Configuração

# %%
import os, re, sys, io, contextlib, warnings
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import networkx as nx
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

def md(texto):
    """Exibe Markdown com vírgula decimal (padrão brasileiro)."""
    display(Markdown(re.sub(r"(?<=\d)\.(?=\d)", ",", texto)))

# Renderizador: Colab no Colab; HTML com plotly via CDN no Jupyter/nbviewer
pio.renderers.default = os.environ.get("PLOTLY_RENDERER", "colab" if IN_COLAB else "notebook_connected")

# ---- Identidade visual (paleta categórica validada para daltonismo, ordem fixa) ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]        # vermelho ↔ cinza ↔ azul
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

pio.templates["estudo"] = go.layout.Template(
    layout=dict(
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED)),
        legend=dict(font=dict(color=INK2)), hoverlabel=dict(font=dict(family="system-ui")),
        margin=dict(l=60, r=30, t=70, b=50),
    )
)
pio.templates.default = "plotly_white+estudo"
print("Ambiente:", "Google Colab" if IN_COLAB else "Jupyter local", "| pandas", pd.__version__)

# %% [markdown]
# ### 0.1 Carga dos dados
# Os arquivos ficam na pasta `data/` do repositório. Se o notebook estiver rodando fora dela (como no Colab), eles são baixados direto do GitHub.

# %%
REPO_RAW = "https://raw.githubusercontent.com/joaogabrielts0611-ai/analise-precos-eua-bls/main/data/"

def carregar(nome, **kw):
    local = os.path.join("data", nome)
    return pd.read_csv(local if os.path.exists(local) else REPO_RAW + nome, **kw)

longo   = carregar("us_average_prices_monthly.csv", parse_dates=["date"])
largo   = carregar("us_average_prices_wide.csv", parse_dates=["date"]).set_index("date").sort_index()
resumo  = carregar("us_average_prices_item_summary.csv", parse_dates=["first_date", "last_date"])
# CPI-U (todos os itens, média das cidades, sem ajuste sazonal, 1982-84=100), série FRED CPIAUCNS
cpi = carregar("cpi_u_nsa_CPIAUCNS.csv", parse_dates=["date"]).set_index("date")["cpi_u"]

print(f"Longo: {longo.shape} | Largo: {largo.shape} | Resumo: {resumo.shape} | CPI: {cpi.notna().sum()} meses")
longo.head()

# %% [markdown]
# ### 0.2 Dicionário de itens: nome em português, grupo e família
#
# * **Grupo** (8): agregação das 12 categorias originais, já com as correções de classificação da Seção 1. É o que dá a cor nos gráficos.
# * **Família**: itens que são praticamente o *mesmo produto* (ex.: as quatro gasolinas, ou "carne moída" e "todas as carnes moídas"). Pares da mesma família são correlacionados por construção, então o ranking de "correlações interessantes" os exclui.

# %%
ITENS = {
 # item original: (rótulo PT, grupo, família)
 "Coffee, 100%, ground roast, all sizes": ("Café moído", "Bebidas e café", "café"),
 "Orange juice, frozen concentrate, 12 oz. can": ("Suco de laranja concentrado", "Hortifrúti", "laranja"),
 "Utility (piped) gas": ("Gás encanado (therm)", "Energia", "gás"),
 "Lettuce, romaine": ("Alface romana", "Hortifrúti", "alface"),
 "All Uncooked Other Beef (Excluding Veal)": ("Outras carnes bovinas (agreg.)", "Carne bovina", "bov_outros"),
 "All soft drinks, 12 pk, 12 oz., cans": ("Refrigerante lata (12 pk)", "Bebidas e café", "refri"),
 "Steak, sirloin, USDA Choice, boneless": ("Contrafilé (sirloin)", "Carne bovina", "bov_bife"),
 "Fuel oil #2": ("Óleo combustível nº 2", "Energia", "destilados"),
 "Chuck roast, USDA Choice, boneless": ("Acém (assado)", "Carne bovina", "bov_assado"),
 "All Uncooked Beef Steaks": ("Bifes bovinos (agreg.)", "Carne bovina", "bov_bife"),
 "Automotive diesel fuel": ("Diesel", "Energia", "destilados"),
 "Ground beef, 100% beef": ("Carne moída", "Carne bovina", "bov_moida"),
 "Gasoline, unleaded premium": ("Gasolina premium", "Energia", "gasolina"),
 "Steak, round, USDA Choice, boneless": ("Bife de patinho (round)", "Carne bovina", "bov_bife"),
 "Ground chuck, 100% beef": ("Acém moído", "Carne bovina", "bov_moida"),
 "Rice, white, long grain, uncooked": ("Arroz branco", "Grãos, padaria e mercearia", "arroz"),
 "All Uncooked Beef Roasts": ("Assados bovinos (agreg.)", "Carne bovina", "bov_assado"),
 "Beef for stew, boneless": ("Carne para ensopado", "Carne bovina", "bov_assado"),
 "Sugar, white, all sizes": ("Açúcar", "Grãos, padaria e mercearia", "açúcar"),
 "Cookies, chocolate chip": ("Cookies", "Grãos, padaria e mercearia", "cookies"),
 "Round roast, USDA Choice, boneless": ("Patinho (assado)", "Carne bovina", "bov_assado"),
 "Potato chips": ("Batata chips", "Grãos, padaria e mercearia", "chips"),
 "Gasoline, unleaded midgrade": ("Gasolina aditivada", "Energia", "gasolina"),
 "Gasoline, all types": ("Gasolina (todos os tipos)", "Energia", "gasolina"),
 "All uncooked ground beef": ("Carnes moídas (agreg.)", "Carne bovina", "bov_moida"),
 "Grapefruit": ("Toranja", "Hortifrúti", "toranja"),
 "Lettuce, iceberg": ("Alface americana", "Hortifrúti", "alface"),
 "Gasoline, unleaded regular": ("Gasolina comum", "Energia", "gasolina"),
 "All soft drinks": ("Refrigerante 2 L", "Bebidas e café", "refri"),
 "All Ham (Excluding Canned Ham and Luncheon Slices)": ("Presuntos (agreg.)", "Suínos e frios", "presunto"),
 "Yogurt": ("Iogurte", "Laticínios", "iogurte"),
 "Potatoes, white": ("Batata", "Hortifrúti", "batata"),
 "Malt beverages, all types, all sizes, any origin": ("Cerveja", "Bebidas e café", "cerveja"),
 "Electricity": ("Eletricidade (kWh)", "Energia", "eletricidade"),
 "Chicken, fresh, whole": ("Frango inteiro", "Aves e ovos", "frango"),
 "Ground beef, lean and extra lean": ("Carne moída magra", "Carne bovina", "bov_moida"),
 "Milk, fresh, low-fat, reduced fat, skim": ("Leite semidesnatado/desnatado", "Laticínios", "leite"),
 "Ham, boneless, excluding canned": ("Presunto sem osso", "Suínos e frios", "presunto"),
 "Ice cream, prepackaged, bulk, regular": ("Sorvete", "Laticínios", "sorvete"),
 "All Other Pork (Excluding Canned Ham and Luncheon Slices)": ("Outros suínos (agreg.)", "Suínos e frios", "suino_outros"),
 "Bread, white, pan": ("Pão de forma branco", "Grãos, padaria e mercearia", "pão"),
 "Bacon, sliced": ("Bacon", "Suínos e frios", "bacon"),
 "Chicken breast, boneless": ("Peito de frango", "Aves e ovos", "frango"),
 "Milk, fresh, whole, fortified": ("Leite integral", "Laticínios", "leite"),
 "Strawberries, dry pint": ("Morango", "Hortifrúti", "morango"),
 "Wine, red and white table, all sizes, any origin": ("Vinho de mesa", "Bebidas e café", "vinho"),
 "Bananas": ("Banana", "Hortifrúti", "banana"),
 "Beans, dried, any type, all sizes": ("Feijão seco", "Grãos, padaria e mercearia", "feijão"),
 "Tomatoes, field grown": ("Tomate", "Hortifrúti", "tomate"),
 "All Pork Chops": ("Costeletas suínas (agreg.)", "Suínos e frios", "costeleta"),
 "Cheddar cheese, natural": ("Queijo cheddar", "Laticínios", "queijo"),
 "Chicken legs, bone-in": ("Coxa de frango", "Aves e ovos", "frango"),
 "Eggs, grade A, large": ("Ovos (dúzia)", "Aves e ovos", "ovos"),
 "American processed cheese": ("Queijo processado", "Laticínios", "queijo"),
 "Flour, white, all purpose": ("Farinha de trigo", "Grãos, padaria e mercearia", "farinha"),
 "Chops, boneless": ("Costeleta suína s/ osso", "Suínos e frios", "costeleta"),
 "Butter, stick": ("Manteiga", "Laticínios", "manteiga"),
 "Corn, canned, any style, all sizes": ("Milho em lata", "Hortifrúti", "milho"),
 "Lemons": ("Limão siciliano", "Hortifrúti", "limão"),
 "Spaghetti and macaroni": ("Macarrão", "Grãos, padaria e mercearia", "macarrão"),
 "Bologna, all beef or mixed": ("Mortadela (bologna)", "Suínos e frios", "frios"),
 "Bread, whole wheat, pan": ("Pão de forma integral", "Grãos, padaria e mercearia", "pão"),
 "Broccoli": ("Brócolis", "Hortifrúti", "brócolis"),
 "Chops, center cut, bone-in": ("Costeleta suína c/ osso", "Suínos e frios", "costeleta"),
 "Chuck roast, graded and ungraded, excluding USDA Prime and Choice": ("Acém (outras classif.)", "Carne bovina", "bov_assado"),
 "Frankfurters, all meat or all beef": ("Salsicha", "Suínos e frios", "frios"),
 "Grapes, Thompson Seedless": ("Uva sem semente", "Hortifrúti", "uva"),
 "Margarine, soft, tubs": ("Margarina", "Laticínios", "margarina"),
 "Oranges, Navel": ("Laranja", "Hortifrúti", "laranja"),
 "Pears, Anjou": ("Pera", "Hortifrúti", "pera"),
 "Peppers, sweet": ("Pimentão", "Hortifrúti", "pimentão"),
 "Steak, round, graded and ungraded, excluding USDA Prime and Choice": ("Patinho (outras classif.)", "Carne bovina", "bov_bife"),
 "Sugar, white, 33-80 oz. pkg": ("Açúcar (pacote 33–80 oz)", "Grãos, padaria e mercearia", "açúcar"),
 "Turkey, frozen, whole": ("Peru congelado", "Aves e ovos", "peru"),
}
GRUPOS = ["Carne bovina", "Energia", "Hortifrúti", "Laticínios", "Grãos, padaria e mercearia",
          "Aves e ovos", "Suínos e frios", "Bebidas e café"]
COR_GRUPO = dict(zip(GRUPOS, SERIES))

assert set(ITENS) == set(resumo["item"]), "Dicionário de itens incompleto"
ROT   = {k: v[0] for k, v in ITENS.items()}
GRUPO = {k: v[1] for k, v in ITENS.items()}
FAM   = {k: v[2] for k, v in ITENS.items()}
print(f"{len(ITENS)} itens mapeados em {len(GRUPOS)} grupos e {len(set(FAM.values()))} famílias.")

# %% [markdown]
# ---
# ## 1. Auditoria da base: o README está certo?
#
# Antes de analisar, vale checar a própria base. Encontramos seis pontos. Os quatro primeiros mudam números do README.

# %% [markdown]
# ### 1.1 Integridade básica

# %%
checks = {
    "Linhas duplicadas (item × mês)": int(longo.duplicated(["item", "date"]).sum()),
    "Preços ausentes": int(longo["price"].isna().sum()),
    "Preços ≤ 0": int((longo["price"] <= 0).sum()),
    "Meses distintos": longo["date"].nunique(),
    "Itens": longo["item"].nunique(),
    "Largo e longo batem?": bool(np.allclose(
        longo.pivot(index="date", columns="item", values="price").reindex(largo.index)[largo.columns].values,
        largo.values, equal_nan=True)),
}
pd.Series(checks, name="resultado").to_frame()

# %% [markdown]
# ### 1.2 Cobertura: buracos nas séries
# Cada linha é um item, cada coluna um mês. Cinza = sem dado. Há três tipos de buraco: (a) séries que começam só em abr/2018; (b) interrupções longas em 2020–2023 (alface americana, toranja, milho em lata, café); (c) **out/2025 vazio para quase todos os itens**, consequência da paralisação do governo federal americano naquele mês, quando o BLS não coletou preços.

# %%
pres = largo.notna().astype(int)
ordem = resumo.sort_values(["is_current", "n_months"], ascending=[False, False])["item"]
pres = pres[ordem]
fig = go.Figure(go.Heatmap(
    z=pres.T.values, x=pres.index, y=[ROT[i] for i in pres.columns],
    colorscale=[[0, "#e1e0d9"], [1, "#2a78d6"]], showscale=False,
    hovertemplate="%{y}<br>%{x|%m/%Y}<extra></extra>"))
fig.update_layout(title="Disponibilidade mensal por item (azul = há preço publicado)",
                  height=1250, yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
fig.show()

out25 = largo.loc["2025-10-01"].notna().sum()
print(f"Itens com preço em out/2025: {out25} de {largo.shape[1]}")

# %% [markdown]
# ### 1.3 Problema 1: classificação errada de categorias
# * **Café** está na categoria `Beef` (carne bovina). Isso inflava a mediana da carne bovina com o maior aumento da base (+96%).
# * **Batata chips** está em `Vegetables` (hortaliças).
# * `Meat, broad category` mistura agregados de carne bovina e presunto.
#
# No dicionário da Seção 0.2 esses itens já foram realocados (café → *Bebidas e café*; chips → *Grãos, padaria e mercearia*; agregados → seus grupos de carne).

# %%
display(resumo.loc[resumo["item"].isin(["Coffee, 100%, ground roast, all sizes", "Potato chips"]) |
                   (resumo["category"] == "Meat, broad category"),
                   ["item", "category"]].assign(grupo_corrigido=lambda d: d["item"].map(GRUPO)))

# %% [markdown]
# ### 1.4 Problema 2: "primeiros 12 meses" não são 12 meses de calendário
# O resumo calcula `avg_first_12mo` como a média das **primeiras 12 observações**, não dos primeiros 12 meses. Quando a série tem buracos, essa "janela" se estende por anos. O caso extremo é o milho em lata: a "média inicial" mistura 4 meses de 2015 com 8 meses de 2023–2024, então os +4,8% do README não comparam 2015 com 2026.

# %%
def janela_obs(item, n=12, inicio=True):
    d = longo.loc[longo["item"] == item].sort_values("date")
    d = d.head(n) if inicio else d.tail(n)
    return d["date"].min().strftime("%Y-%m"), d["date"].max().strftime("%Y-%m"), d["price"].mean()

linhas = []
for it in resumo.loc[resumo["is_current"], "item"]:
    a0, a1, _ = janela_obs(it, inicio=True)
    b0, b1, _ = janela_obs(it, inicio=False)
    meses_a = (pd.Period(a1, "M") - pd.Period(a0, "M")).n + 1
    meses_b = (pd.Period(b1, "M") - pd.Period(b0, "M")).n + 1
    if meses_a > 13 or meses_b > 13:
        linhas.append((ROT[it], f"{a0} → {a1}", meses_a, f"{b0} → {b1}", meses_b))
pd.DataFrame(linhas, columns=["item", "janela inicial usada", "meses cobertos", "janela final usada", "meses cobertos "])

# %% [markdown]
# ### 1.5 Problema 3: bases diferentes e a mediana de +44%
# Cinco séries só começam em **abr/2018** (leite semidesnatado, iogurte, manteiga, dois refrigerantes). O resumo compara 2018 → 2026 para elas e 2015 → 2026 para as demais, e depois tira uma mediana única. Não é a mesma comparação. A correção: comparar todos na **mesma janela de calendário** e reportar também a **taxa anual equivalente**, que neutraliza horizontes diferentes.
#
# ### 1.6 Problema 4: `is_current` descarta séries que estão vivas
# A flag marca como "descontinuado" qualquer série sem preço em jul/2026. Mas pão integral, costeleta com osso e laranja têm dados até abr–jun/2026, e o patinho "outras classificações" voltou a ser publicado em 2024. Elas têm dados suficientes na janela final.
#
# ### 1.7 Problema 5: a inflação do período foi maior que 35%
# Calculando com o CPI-U oficial (série FRED `CPIAUCNS`, sem ajuste sazonal, arquivo `data/cpi_u_nsa_CPIAUCNS.csv`) e as mesmas janelas de 12 meses:

# %%
cpi_m = cpi.interpolate(limit_area="inside")  # out/2025 não foi publicado: interpolado
JAN_A = ("2015-01-01", "2015-12-01")   # janela inicial: ano-calendário 2015
JAN_B = ("2025-08-01", "2026-07-01")   # janela final: últimos 12 meses da base
inf_total = cpi_m[JAN_B[0]:JAN_B[1]].mean() / cpi_m[JAN_A[0]:JAN_A[1]].mean() - 1
print(f"CPI-U médio 2015: {cpi_m[JAN_A[0]:JAN_A[1]].mean():.1f}")
print(f"CPI-U médio ago/2025–jul/2026: {cpi_m[JAN_B[0]:JAN_B[1]].mean():.1f}")
print(f"Inflação acumulada entre as janelas: {inf_total:+.1%}  (README: ~35%)")
print(f"Taxa anual equivalente: {(1+inf_total)**(1/10.583)-1:+.2%} a.a.")

# %% [markdown]
# Com 38,5% em vez de 35%, alguns itens que pareciam "empatar com a inflação" na verdade **ficaram mais baratos em termos reais**.
#
# ### 1.8 Problema 6: o período-base dos ovos estava contaminado
# A gripe aviária de **2015** (a maior epidemia da história dos EUA até então) fez o preço dos ovos disparar no segundo semestre daquele ano. Como 2015 é a base, a variação dos ovos parece pequena (+9,5%). Detalhado na Seção 2.6.

# %% [markdown]
# ### 1.9 Base corrigida
# Construímos uma tabela-mestre com:
# * preços **nominais** e **reais** (em US$ de jul/2026, deflacionados pelo CPI-U);
# * buracos de **até 2 meses** dentro da série preenchidos por interpolação do log do preço (cobre out/2025 sem inventar tendências); buracos longos ficam vazios;
# * janela inicial = ano de 2015 (ou os 12 primeiros meses de calendário, se a série começa depois), janela final = ago/2025–jul/2026. Exigimos ao menos 8 meses com dado em cada janela.

# %%
datas = largo.index
P = largo.copy()                                                       # nominal, com buracos
logP = np.log(P).interpolate(limit=2, limit_area="inside")             # buracos curtos preenchidos
Pf = np.exp(logP)                                                      # nominal "preenchido"
deflator = cpi_m.reindex(datas)
R = Pf.mul(deflator.loc["2026-07-01"] / deflator, axis=0)              # real, US$ de jul/2026

def janela_inicial(serie):
    s = serie.dropna()
    ini = s.index.min()
    if ini <= pd.Timestamp("2015-01-01"):
        return JAN_A
    return (ini.strftime("%Y-%m-01"), (ini + pd.DateOffset(months=11)).strftime("%Y-%m-01"))

reg = []
for it in largo.columns:
    a = janela_inicial(P[it])
    pa, pb = P[it][a[0]:a[1]], P[it][JAN_B[0]:JAN_B[1]]
    ra, rb = R[it][a[0]:a[1]].where(pa.notna()), R[it][JAN_B[0]:JAN_B[1]].where(pb.notna())
    anos = (pd.Timestamp(JAN_B[0]) - pd.Timestamp(a[0])).days / 365.25
    ok = pa.notna().sum() >= 8 and pb.notna().sum() >= 8
    nom = pb.mean() / pa.mean() - 1 if ok else np.nan
    real = rb.mean() / ra.mean() - 1 if ok else np.nan
    reg.append(dict(item=it, rotulo=ROT[it], grupo=GRUPO[it], familia=FAM[it],
                    unidade=resumo.set_index("item").loc[it, "unit"],
                    base=f"{a[0][:7]}", meses_base=int(pa.notna().sum()), meses_final=int(pb.notna().sum()),
                    ativo=ok, preco_base=pa.mean(), preco_final=pb.mean(),
                    var_nominal=nom, var_real=real,
                    taxa_nominal_aa=(1 + nom) ** (1 / anos) - 1 if ok else np.nan,
                    taxa_real_aa=(1 + real) ** (1 / anos) - 1 if ok else np.nan,
                    var_readme=resumo.set_index("item").loc[it, "pct_change_12mo_avg"] / 100))
T = pd.DataFrame(reg).set_index("item")
ATIVOS = T.index[T["ativo"]].tolist()
print(f"Séries com dados suficientes nas duas janelas: {len(ATIVOS)} (README: 60)")
print("Recuperadas pela correção:", ", ".join(T.loc[T['ativo'] & T['var_readme'].isna(), 'rotulo']))
print("Excluídas pela correção:  ", ", ".join(T.loc[~T['ativo'] & T['var_readme'].notna(), 'rotulo']) or "nenhuma")

# %% [markdown]
# #### Comparação: README × recalculado
# As diferenças maiores aparecem justamente nos itens com buracos ou base diferente.

# %%
cmp_ = T.loc[T["ativo"] & T["var_readme"].notna(), ["rotulo", "base", "var_readme", "var_nominal"]].copy()
cmp_["diferença (p.p.)"] = (cmp_["var_nominal"] - cmp_["var_readme"]) * 100
cmp_.sort_values("diferença (p.p.)", key=abs, ascending=False).head(10).style.format(
    {"var_readme": "{:+.1%}", "var_nominal": "{:+.1%}", "diferença (p.p.)": "{:+.1f}"}, decimal=",").hide(axis="index")

# %% [markdown]
# ---
# ## 2. Análise inicial aprimorada

# %% [markdown]
# ### 2.1 Quem ganhou da inflação? Variação nominal × real
# Com a base corrigida, todos os itens ativos continuam mais caros em dólares correntes. Mas descontada a inflação, a história muda: parte relevante ficou **mais barata em termos reais**.

# %%
A = T.loc[ATIVOS].copy()
n_sub = int((A["var_nominal"] > 0).sum()); n_real_pos = int((A["var_real"] > 0).sum())
md((
    f"* **{n_sub} de {len(A)}** itens subiram em termos nominais; mediana **{A['var_nominal'].median():+.1%}**.\n"
    f"* Em termos reais, só **{n_real_pos} de {len(A)}** subiram acima da inflação; mediana real **{A['var_real'].median():+.1%}**.\n"
    f"* **{len(A) - n_real_pos} itens ficaram mais baratos** que a cesta média do consumidor americano."))

A_ord = A.sort_values("var_real")
fig = go.Figure()
for g in GRUPOS:
    d = A_ord[A_ord["grupo"] == g]
    fig.add_bar(y=d["rotulo"], x=d["var_real"] * 100, orientation="h", name=g, marker_color=COR_GRUPO[g],
                customdata=np.c_[d["var_nominal"] * 100, d["base"]],
                hovertemplate="<b>%{y}</b><br>Real: %{x:+.1f}%<br>Nominal: %{customdata[0]:+.1f}%<br>Base: %{customdata[1]}<extra></extra>")
fig.add_vline(x=0, line_color=INK2, line_width=1)
fig.update_layout(title=f"Variação real do preço (descontado o CPI-U de {inf_total:+.1%}), 2015 → ago/25–jul/26",
                  xaxis_title="Variação real (%)", height=1300, barmode="overlay",
                  yaxis=dict(categoryorder="array", categoryarray=A_ord["rotulo"].tolist(), tickfont=dict(size=11)),
                  legend=dict(orientation="h", y=1.005, x=0, yanchor="bottom"), bargap=0.25, margin=dict(t=150))
fig.show()

# %% [markdown]
# ### 2.2 Por grupo: mediana nominal, real e taxa anual
# A taxa anual (a.a.) permite comparar itens com bases em 2015 e em 2018.

# %%
tab_g = (A.groupby("grupo")
         .agg(itens=("rotulo", "size"), mediana_nominal=("var_nominal", "median"),
              mediana_real=("var_real", "median"), taxa_real_aa=("taxa_real_aa", "median"),
              acima_inflacao=("var_real", lambda s: f"{(s > 0).sum()}/{len(s)}"))
         .sort_values("mediana_real", ascending=False))
tab_g.style.format({"mediana_nominal": "{:+.1%}", "mediana_real": "{:+.1%}", "taxa_real_aa": "{:+.2%}"}, decimal=",")

# %% [markdown]
# ### 2.3 Índices por grupo (base 2015 = 100)
# Cada índice é a **média geométrica** dos preços relativos dos itens do grupo (cada item dividido pela sua própria média de 2015), usando apenas itens com série desde 2015. É o mesmo princípio dos índices elementares do CPI. Clique na legenda para isolar grupos.

# %%
base15 = [i for i in ATIVOS if T.loc[i, "base"] == "2015-01"]
rel = np.log(Pf[base15] / Pf[base15].loc["2015"].mean())

def indice_grupo(itens):
    sub = rel[itens]
    return 100 * np.exp(sub.mean(axis=1, skipna=True).where(sub.notna().sum(axis=1) >= max(1, len(itens) // 2)))

IDX = pd.DataFrame({g: indice_grupo([i for i in base15 if GRUPO[i] == g]) for g in GRUPOS})
IDX["Alimentos (todos)"] = indice_grupo([i for i in base15 if GRUPO[i] != "Energia"])  # alimentos e bebidas
IDX["CPI-U"] = 100 * deflator / deflator.loc["2015"].mean()

fig = go.Figure()
for g in GRUPOS:
    fig.add_scatter(x=IDX.index, y=IDX[g], name=g, line=dict(width=2, color=COR_GRUPO[g]),
                    hovertemplate=f"{g}<br>%{{x|%m/%Y}}: %{{y:.1f}}<extra></extra>")
fig.add_scatter(x=IDX.index, y=IDX["CPI-U"], name="CPI-U (inflação geral)",
                line=dict(width=3, color=INK, dash="dash"), hovertemplate="CPI-U<br>%{x|%m/%Y}: %{y:.1f}<extra></extra>")
fig.update_layout(title="Índices de preço por grupo (média 2015 = 100)", yaxis_title="Índice", height=520,
                  hovermode="x unified")
fig.show()

# %% [markdown]
# ### 2.4 Regimes: antes, durante e depois do choque
# Taxa anual média (log) de variação real em três períodos. Mostra que a maior parte do aumento real está concentrada em 2020–2022, e que em vários grupos houve devolução depois.

# %%
REGIMES = {"Pré-pandemia (2015–2019)": ("2015-01-01", "2019-12-01"),
           "Choque (2020–2022)": ("2020-01-01", "2022-12-01"),
           "Pós-choque (2023–jul/2026)": ("2023-01-01", "2026-07-01")}
IDXr = IDX[GRUPOS + ["Alimentos (todos)"]].div(IDX["CPI-U"], axis=0) * 100
lin = []
for nome, (a, b) in REGIMES.items():
    anos = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    for g in IDXr.columns:
        lin.append(dict(regime=nome, grupo=g, taxa=100 * (np.log(IDXr.loc[b, g]) - np.log(IDXr.loc[a, g])) / anos))
reg_df = pd.DataFrame(lin)
piv = reg_df.pivot(index="grupo", columns="regime", values="taxa")[list(REGIMES)]
fig = px.imshow(piv, text_auto=".1f", color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                aspect="auto", labels=dict(color="% a.a. real"))
fig.update_layout(title="Variação real anualizada por regime (% a.a., acima/abaixo do CPI-U)", height=460)
fig.update_traces(hovertemplate="%{y}<br>%{x}<br>%{z:+.2f}% a.a.<extra></extra>")
fig.show()

# %% [markdown]
# ### 2.5 Risco × retorno: volatilidade e variação real
# Para um consumidor (ou um comprador de empresa), não importa só *quanto* um preço sobe, mas *quão previsível* ele é. Volatilidade = desvio-padrão anualizado da variação mensal do log do preço, **depois de remover a sazonalidade** (média de cada mês do ano).

# %%
dlog = logP.diff()
dlog_ds = dlog - dlog.groupby(dlog.index.month).transform("mean")    # variação mensal dessazonalizada
A["volatilidade_aa"] = dlog_ds[ATIVOS].std() * np.sqrt(12)

fig = go.Figure()
for g in GRUPOS:
    d = A[A["grupo"] == g]
    fig.add_scatter(x=d["volatilidade_aa"] * 100, y=d["taxa_real_aa"] * 100, mode="markers", name=g,
                    marker=dict(size=11, color=COR_GRUPO[g], line=dict(width=2, color=SURFACE)),
                    text=d["rotulo"], hovertemplate="<b>%{text}</b><br>Volatilidade: %{x:.1f}% a.a.<br>Variação real: %{y:+.2f}% a.a.<extra></extra>")
for it in A.nlargest(4, "volatilidade_aa").index.tolist() + A.nlargest(2, "taxa_real_aa").index.tolist():
    fig.add_annotation(x=A.loc[it, "volatilidade_aa"] * 100, y=A.loc[it, "taxa_real_aa"] * 100, text=A.loc[it, "rotulo"],
                       showarrow=False, xanchor="left", xshift=9, font=dict(size=11, color=INK2))
fig.add_hline(y=0, line_color=AXIS)
fig.update_layout(title="Volatilidade × variação real anual (cada ponto é um item)",
                  xaxis_title="Volatilidade (% a.a., sem sazonalidade)", yaxis_title="Variação real (% a.a.)", height=560)
fig.show()

# %% [markdown]
# ### 2.6 Sazonalidade: quais preços têm "época"?
# Perfil sazonal = desvio médio do log do preço em relação à sua média móvel centrada de 12 meses, por mês do ano. A amplitude (pico − vale) mede quanto do preço de um mês é só calendário. É por isso que comparar um mês isolado com outro engana.

# %%
mm12 = logP.rolling(12, center=True, min_periods=10).mean()
desvio = (logP - mm12)
perfil = desvio.groupby(desvio.index.month).mean()
amp = (perfil.max() - perfil.min()).loc[ATIVOS].sort_values(ascending=False)
top_saz = amp.head(15).index
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
z = (perfil[top_saz].T * 100)
fig = px.imshow(z.values, x=MESES, y=[f"{ROT[i]} ({amp[i]*100:.0f}%)" for i in top_saz],
                color_continuous_scale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                color_continuous_midpoint=0, aspect="auto", labels=dict(color="% vs. tendência"))
fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:+.1f}% em relação à tendência<extra></extra>")
fig.update_layout(title="Os 15 itens mais sazonais (vermelho = mês caro, azul = mês barato; amplitude entre parênteses)", height=560)
fig.show()

# %% [markdown]
# ### 2.7 Ovos: o degrau ficou?
# A pergunta do README: depois do pico de mar/2025 (US$ 6,23), o preço voltou à tendência? Ajustamos uma tendência log-linear no período "calmo" (2016–2019, depois do surto de 2015 e antes da pandemia) e projetamos para frente, em termos nominais e reais.

# %%
ovo = "Eggs, grade A, large"
calmo = logP.loc["2016-01-01":"2019-12-01", ovo]
t_calmo = (calmo.index - datas[0]).days.values / 365.25
b1, b0 = np.polyfit(t_calmo, calmo.values, 1)
t_all = (datas - datas[0]).days.values / 365.25
tend = np.exp(b0 + b1 * t_all)
# tendência alternativa: preço médio 2016–2019 corrigido pela inflação (CPI)
tend_cpi = P.loc["2016":"2019", ovo].mean() * deflator / deflator.loc["2016":"2019"].mean()

fig = go.Figure()
fig.add_scatter(x=datas, y=P[ovo], name="Preço observado", line=dict(color=SERIES[0], width=2),
                hovertemplate="%{x|%m/%Y}: US$ %{y:.2f}<extra></extra>")
fig.add_scatter(x=datas, y=tend_cpi, name="Média 2016–19 corrigida pelo CPI", line=dict(color=INK2, width=2, dash="dash"),
                hovertemplate="%{x|%m/%Y}: US$ %{y:.2f}<extra></extra>")
for x0, x1, txt in [("2015-04-01", "2015-12-01", "Gripe aviária 2015"), ("2022-02-01", "2023-03-01", "Gripe aviária 2022–23"),
                    ("2024-11-01", "2025-05-01", "Pico 2025")]:
    fig.add_vrect(x0=x0, x1=x1, fillcolor="#e34948", opacity=0.08, line_width=0, layer="below",
                  annotation_text=txt, annotation_position="top left", annotation_font=dict(size=11, color=INK2))
fig.update_layout(title="Ovos grade A, dúzia: preço mensal e tendência pré-crise", yaxis_title="US$ por dúzia", height=480,
                  hovermode="x unified")
fig.show()

ult = P.loc[JAN_B[0]:JAN_B[1], ovo].mean()
md((
    f"* Média dos últimos 12 meses: **US$ {ult:.2f}**; tendência pré-crise corrigida pela inflação: **US$ {tend_cpi.loc[JAN_B[0]:JAN_B[1]].mean():.2f}** "
    f"(**{ult / tend_cpi.loc[JAN_B[0]:JAN_B[1]].mean() - 1:+.0%}**).\n"
    f"* Base 2015 (com surto): US$ {P.loc['2015', ovo].mean():.2f}; base 2016–2019 (calma): US$ {P.loc['2016':'2019', ovo].mean():.2f}. "
    f"Medido contra 2016–2019, o aumento nominal é **{ult / P.loc['2016':'2019', ovo].mean() - 1:+.0%}**, não +9,5%.\n"
    f"* Último mês (jul/2026): US$ {P[ovo].iloc[-1]:.2f}. Mínimo da série: US$ {P[ovo].min():.2f} ({P[ovo].idxmin():%m/%Y}); "
    f"máximo: US$ {P[ovo].max():.2f} ({P[ovo].idxmax():%m/%Y})."))

# %% [markdown]
# ### 2.8 O choque de 2022: em que ordem os preços reagiram?
# Para cada grupo, o mês em que a inflação em 12 meses (variação do índice contra o mesmo mês do ano anterior) atingiu o pico entre 2021 e 2024. As carnes chegam ao pico primeiro (fim de 2021, efeito de oferta concentrada nos frigoríficos); a energia em meados de 2022; e os alimentos mais processados (padaria, grãos, laticínios) por último, no fim de 2022. Itens com mais etapas de processamento e contratos mais longos repassam custos com mais atraso.

# %%
yoy_idx = np.log(IDX).diff(12) * 100
pk = []
for g in GRUPOS + ["Alimentos (todos)", "CPI-U"]:
    s = yoy_idx.loc["2021-01-01":"2024-12-01", g].dropna()
    pk.append(dict(grupo=g, mes_pico=s.idxmax(), pico=s.max()))
pk = pd.DataFrame(pk).sort_values("mes_pico")
cores_pk = [COR_GRUPO.get(g, INK) for g in pk["grupo"]]
fig = go.Figure(go.Scatter(x=pk["mes_pico"], y=pk["grupo"], mode="markers+text",
                           marker=dict(size=pk["pico"].clip(lower=4) * 0.9 + 6, color=cores_pk, line=dict(width=2, color=SURFACE)),
                           text=[f"{v:+.0f}%" for v in pk["pico"]], textposition="middle right",
                           textfont=dict(color=INK2, size=11),
                           hovertemplate="%{y}<br>Pico em %{x|%m/%Y}<br>%{text} em 12 meses<extra></extra>"))
fig.update_layout(title="Mês do pico da inflação em 12 meses, por grupo (tamanho = intensidade)", height=460,
                  xaxis=dict(range=["2021-07-01", "2023-06-01"], tickformat="%m/%Y", dtick="M3"), yaxis=dict(autorange="reversed"))
fig.show()

# %% [markdown]
# ---
# ## 3. Correlações entre pares
#
# ### 3.1 A armadilha: correlação de níveis é quase sempre alta e quase sempre espúria
# Quase todos os preços sobem com o tempo. Duas séries com tendência têm correlação alta mesmo sem relação nenhuma entre si (correlação espúria, Granger & Newbold, 1974). Por isso comparamos três medidas para cada um dos pares de itens ativos:
#
# | Medida | O que captura | Uso |
# |---|---|---|
# | **Nível** (preço × preço) | tendência comum | só ilustração, **não** usar para inferência |
# | **Variação em 12 meses** | ciclos de médio prazo, sem sazonalidade | leitura econômica; p-valores inflados (janelas se sobrepõem) |
# | **Variação mensal dessazonalizada** | choques de curto prazo em comum | **medida principal** para testes |

# %%
def corr_pares(M, min_n=60):
    """Tabela longa de correlações de Pearson entre todos os pares, com n e p-valor."""
    cols = M.columns
    X = M.values
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            m = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            n = int(m.sum())
            if n < min_n:
                continue
            r, p = stats.pearsonr(X[m, i], X[m, j])
            out.append((cols[i], cols[j], r, n, p))
    d = pd.DataFrame(out, columns=["a", "b", "r", "n", "p"])
    d["rotulo_a"], d["rotulo_b"] = d["a"].map(ROT), d["b"].map(ROT)
    d["grupo_a"], d["grupo_b"] = d["a"].map(GRUPO), d["b"].map(GRUPO)
    d["mesma_familia"] = d["a"].map(FAM) == d["b"].map(FAM)
    d["mesmo_grupo"] = d["grupo_a"] == d["grupo_b"]
    d["p_fdr"] = multipletests(d["p"], method="fdr_bh")[1]
    return d

TRANSF = {
    "Nível": Pf[ATIVOS],
    "Variação em 12 meses": logP[ATIVOS].diff(12),
    "Variação mensal dessazonalizada": dlog_ds[ATIVOS],
}
PARES = {k: corr_pares(v) for k, v in TRANSF.items()}
res = pd.DataFrame({k: dict(pares=len(d), r_medio=d["r"].mean(), r_mediano=d["r"].median(),
                            frac_r_maior_05=(d["r"].abs() > 0.5).mean(),
                            frac_signif_fdr5=(d["p_fdr"] < 0.05).mean()) for k, d in PARES.items()}).T
res.style.format({"pares": "{:,.0f}", "r_medio": "{:.2f}", "r_mediano": "{:.2f}", "frac_r_maior_05": "{:.0%}", "frac_signif_fdr5": "{:.0%}"}, decimal=",", thousands=".")

# %%
fig = go.Figure()
for k, cor in zip(PARES, [SERIES[7], SERIES[3], SERIES[0]]):
    fig.add_histogram(x=PARES[k]["r"], name=k, xbins=dict(start=-1, end=1, size=0.05), marker_color=cor, opacity=0.75)
fig.update_layout(barmode="overlay", title=f"Distribuição das correlações entre os {len(PARES['Nível'])} pares, por medida",
                  xaxis_title="Correlação de Pearson (r)", yaxis_title="Nº de pares", height=460, margin=dict(t=110),
                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
fig.show()

# %% [markdown]
# **Exemplo concreto.** Um par com correlação de nível altíssima e nenhuma relação nos movimentos:

# %%
nv = PARES["Nível"].set_index(["a", "b"])["r"]
dm = PARES["Variação mensal dessazonalizada"].set_index(["a", "b"])["r"]
ambos = pd.concat([nv.rename("nivel"), dm.rename("mensal")], axis=1).dropna()
ambos = ambos[[FAM[a] != FAM[b] and GRUPO[a] != GRUPO[b] for a, b in ambos.index]]
ex = (ambos["nivel"] - ambos["mensal"].abs()).idxmax()
a, b = ex
fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Níveis: r = {ambos.loc[ex,'nivel']:.2f}",
                                                    f"Variações mensais: r = {ambos.loc[ex,'mensal']:.2f}"))
fig.add_scatter(x=Pf[a], y=Pf[b], mode="markers", marker=dict(size=8, color=SERIES[7], opacity=.7), row=1, col=1,
                hovertemplate="%{x:.2f} / %{y:.2f}<extra></extra>", showlegend=False)
fig.add_scatter(x=dlog_ds[a] * 100, y=dlog_ds[b] * 100, mode="markers", marker=dict(size=8, color=SERIES[0], opacity=.7),
                row=1, col=2, hovertemplate="%{x:.1f}% / %{y:.1f}%<extra></extra>", showlegend=False)
fig.update_xaxes(title_text=f"{ROT[a]} (US$)", row=1, col=1); fig.update_yaxes(title_text=f"{ROT[b]} (US$)", row=1, col=1)
fig.update_xaxes(title_text=f"{ROT[a]} (% mês)", row=1, col=2); fig.update_yaxes(title_text=f"{ROT[b]} (% mês)", row=1, col=2)
fig.update_layout(title=f"Correlação espúria: {ROT[a]} × {ROT[b]}", height=430)
fig.show()

# %% [markdown]
# ### 3.2 Matriz de correlação clusterizada (variação mensal dessazonalizada)
# Os itens foram reordenados por **agrupamento hierárquico** (distância = 1 − r, ligação média), de modo que blocos azuis mostram itens que se movem juntos. Passe o mouse para ver o par e o valor.

# %%
C = TRANSF["Variação mensal dessazonalizada"].corr(min_periods=60)
D = (1 - C.fillna(0)).clip(lower=0).to_numpy(copy=True)
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
ordem_cl = C.index[leaves_list(Z)]
Cc = C.loc[ordem_cl, ordem_cl]
fig = go.Figure(go.Heatmap(z=Cc.values, x=[ROT[i] for i in ordem_cl], y=[ROT[i] for i in ordem_cl],
                           colorscale=DIVERGING, zmin=-1, zmax=1, colorbar=dict(title="r"),
                           hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>"))
fig.update_layout(title="Correlação entre variações mensais dessazonalizadas (itens ordenados por cluster)",
                  height=1050, width=1100, xaxis=dict(tickfont=dict(size=9), tickangle=-60),
                  yaxis=dict(tickfont=dict(size=9), autorange="reversed"))
fig.show()

# %% [markdown]
# ### 3.3 Ranking dos pares
# Separamos os pares em **óbvios** (mesma família de produto, como gasolina comum × aditivada) e **não óbvios**. Significância: p-valor corrigido por **Benjamini-Hochberg (FDR)**, necessário porque testar quase 2 mil pares a 5% geraria cerca de 90 "descobertas" por puro acaso. Duas colunas de robustez: a correlação de **Spearman** (por postos, resistente a meses extremos) e o **r sem mar–ago/2020**, que exclui o choque da pandemia. Quando o r cai muito nessas colunas, a correlação depende de poucos meses atípicos.

# %%
def tabela_pares(d, n=15, excluir_familia=True, so_entre_grupos=False):
    x = d.copy()
    if excluir_familia: x = x[~x["mesma_familia"]]
    if so_entre_grupos: x = x[~x["mesmo_grupo"]]
    x = x.reindex(x["r"].abs().sort_values(ascending=False).index).head(n)
    return x

SEM_2020 = ~((TRANSF["Nível"].index >= "2020-03-01") & (TRANSF["Nível"].index <= "2020-08-01"))

def com_spearman(x, M):
    """Acrescenta Spearman e a correlação sem mar–ago/2020 (teste de robustez)."""
    x = x.copy()
    x["spearman"] = [M[[a, b]].dropna().corr(method="spearman").iloc[0, 1] for a, b in zip(x["a"], x["b"])]
    x["r_sem_2020"] = [M.loc[SEM_2020, [a, b]].dropna().corr().iloc[0, 1] for a, b in zip(x["a"], x["b"])]
    return x

def mostrar_pares(x):
    cols = ["rotulo_a", "rotulo_b", "grupo_a", "grupo_b", "r", "spearman", "r_sem_2020", "n", "p_fdr"]
    return (x[cols].rename(columns={"rotulo_a": "item A", "rotulo_b": "item B", "grupo_a": "grupo A", "grupo_b": "grupo B",
                                    "r_sem_2020": "r sem mar–ago/20", "p_fdr": "p (FDR)"})
            .style.format({"r": "{:+.2f}", "spearman": "{:+.2f}", "r sem mar–ago/20": "{:+.2f}", "p (FDR)": "{:.1e}"}, decimal=",")
            .background_gradient(subset=["r"], cmap="RdBu", vmin=-1, vmax=1).hide(axis="index"))

DM = PARES["Variação mensal dessazonalizada"]
md((f"**Pares significativos (FDR < 5%), variação mensal:** {int((DM['p_fdr'] < .05).sum())} de {len(DM)} "
                 f"({(DM['p_fdr'] < .05).mean():.0%}). Entre grupos diferentes: {int(((DM['p_fdr'] < .05) & ~DM['mesmo_grupo']).sum())}."))
md(("#### a) Pares não óbvios mais fortes, dentro ou entre grupos"))
display(mostrar_pares(com_spearman(tabela_pares(DM, 15), TRANSF["Variação mensal dessazonalizada"])))

# %%
md(("#### b) Pares mais fortes **entre grupos diferentes** (variação mensal dessazonalizada)"))
display(mostrar_pares(com_spearman(tabela_pares(DM, 15, so_entre_grupos=True), TRANSF["Variação mensal dessazonalizada"])))
md(("#### c) Pares mais fortes entre grupos diferentes, **variação em 12 meses** (ciclos de médio prazo)"))
Y12 = PARES["Variação em 12 meses"]
display(mostrar_pares(com_spearman(tabela_pares(Y12, 15, so_entre_grupos=True), TRANSF["Variação em 12 meses"])))

# %% [markdown]
# ### 3.4 Correlação entre grupos
# Mesma lógica, agora com os índices de grupo da Seção 2.3. Mostramos lado a lado as duas medidas de variação.

# %%
lidx = np.log(IDX[GRUPOS])
dI = lidx.diff(); dI_ds = dI - dI.groupby(dI.index.month).transform("mean")
CG1 = dI_ds.corr(); CG12 = lidx.diff(12).corr()
fig = make_subplots(rows=1, cols=2, subplot_titles=("Variação mensal dessazonalizada", "Variação em 12 meses"),
                    horizontal_spacing=0.18)
for k, M in enumerate([CG1, CG12], start=1):
    fig.add_heatmap(z=M.values, x=GRUPOS, y=GRUPOS, colorscale=DIVERGING, zmin=-1, zmax=1, text=M.round(2).values,
                    texttemplate="%{text:.2f}", showscale=(k == 2), row=1, col=k,
                    hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>")
fig.update_yaxes(autorange="reversed")
fig.update_layout(title="Correlação entre índices de grupo", height=560, width=1150)
fig.show()

# %% [markdown]
# ### 3.5 Rede de co-movimento
# Cada nó é um item; uma ligação aparece quando o par **não é da mesma família**, a correlação mensal dessazonalizada é ≥ 0,30 e o par é significativo após FDR. Itens isolados se movem por fatores próprios (clima, doenças, safra). O rótulo aparece no item mais conectado de cada grupo; passe o mouse nos demais.

# %%
LIM = 0.30
ed = DM[(~DM["mesma_familia"]) & (DM["r"] >= LIM) & (DM["p_fdr"] < 0.05)]
G = nx.Graph()
G.add_nodes_from(ATIVOS)
G.add_weighted_edges_from(zip(ed["a"], ed["b"], ed["r"]))
pos = nx.spring_layout(G, weight="weight", seed=7, k=0.35)
fig = go.Figure()
ex_, ey_ = [], []
for a_, b_ in G.edges():
    ex_ += [pos[a_][0], pos[b_][0], None]; ey_ += [pos[a_][1], pos[b_][1], None]
fig.add_scatter(x=ex_, y=ey_, mode="lines", line=dict(color=AXIS, width=1), hoverinfo="skip", showlegend=False)
for g in GRUPOS:
    nos = [i for i in ATIVOS if GRUPO[i] == g]
    fig.add_scatter(x=[pos[i][0] for i in nos], y=[pos[i][1] for i in nos], mode="markers+text", name=g,
                    marker=dict(size=[10 + 2.5 * G.degree(i) for i in nos], color=COR_GRUPO[g], line=dict(width=2, color=SURFACE)),
                    text=[ROT[i] if (G.degree(i) > 0 and i == max(nos, key=G.degree)) else "" for i in nos], textposition="top center",
                    textfont=dict(size=10, color=INK2),
                    customdata=[[ROT[i], G.degree(i)] for i in nos],
                    hovertemplate="<b>%{customdata[0]}</b><br>Ligações: %{customdata[1]}<extra></extra>")
fig.update_layout(title=f"Rede de co-movimento (r ≥ {LIM}, FDR < 5%, sem pares da mesma família): {G.number_of_edges()} ligações",
                  height=720, xaxis=dict(visible=False), yaxis=dict(visible=False))
fig.show()
isolados = [ROT[i] for i in ATIVOS if G.degree(i) == 0]
print(f"Itens sem nenhuma ligação ({len(isolados)}):", ", ".join(isolados))

# %% [markdown]
# ### 3.6 Existe um "fator comum"? Análise de componentes principais (PCA)
# Se os preços fossem dirigidos por uma força única (inflação geral, custo de energia), o 1º componente principal explicaria grande parte da variância. Usamos os itens com série completa desde 2015, variação mensal dessazonalizada e padronizada.

# %%
M = TRANSF["Variação mensal dessazonalizada"][base15].loc["2015-02-01":].dropna(axis=1, thresh=130).dropna()
Zs = (M - M.mean()) / M.std()
U, Sv, Vt = np.linalg.svd(Zs.values, full_matrices=False)
expl = Sv ** 2 / (Sv ** 2).sum()
ld = pd.DataFrame(Vt[:3].T, index=M.columns, columns=["PC1", "PC2", "PC3"])
for c_ in ld:  # sinal: maioria dos pesos positiva
    if ld[c_].sum() < 0: ld[c_] *= -1

# referência: variância do 1º componente se as séries fossem independentes (embaralhando cada coluna)
rng = np.random.default_rng(0)
nulo = []
for _ in range(200):
    Xp = np.column_stack([rng.permutation(Zs.values[:, j]) for j in range(Zs.shape[1])])
    s_ = np.linalg.svd(Xp, compute_uv=False); nulo.append((s_[0] ** 2) / (s_ ** 2).sum())
md((f"* {M.shape[1]} itens, {M.shape[0]} meses. **PC1 explica {expl[0]:.1%}** da variância; PC2 {expl[1]:.1%}; PC3 {expl[2]:.1%}.\n"
                 f"* Se os itens fossem independentes, o PC1 explicaria cerca de {np.mean(nulo):.1%} (95º percentil: {np.percentile(nulo,95):.1%}). "
                 f"Existe estrutura comum, mas ela é **fraca**: a maior parte do movimento mensal de cada preço é específica do item."))

fig = make_subplots(rows=1, cols=3, column_widths=[0.3, 0.35, 0.35], horizontal_spacing=0.2,
                    subplot_titles=("Variância explicada (10 primeiros)", "PC1: 10 maiores pesos", "PC2: 10 maiores pesos"))
fig.add_bar(x=[f"PC{i}" for i in range(1, 11)], y=expl[:10] * 100, marker_color=SERIES[0], row=1, col=1,
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>", showlegend=False)
fig.add_hline(y=np.mean(nulo) * 100, line_dash="dot", line_color=INK2, row=1, col=1,
              annotation_text="PC1 se independentes", annotation_font=dict(size=10, color=INK2))
for col, pc in [(2, "PC1"), (3, "PC2")]:
    t_ = ld[pc].reindex(ld[pc].abs().nlargest(10).index)[::-1]
    fig.add_bar(y=[ROT[i] for i in t_.index], x=t_.values, orientation="h", showlegend=False, row=1, col=col,
                marker_color=[COR_GRUPO[GRUPO[i]] for i in t_.index], hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    fig.update_xaxes(title_text="Peso (loading)", row=1, col=col)
fig.update_yaxes(title_text="%", row=1, col=1)
fig.update_layout(title="Análise de componentes principais (cor = grupo do item)", height=480)
fig.show()

# %% [markdown]
# ### 3.7 As correlações mudam com o tempo?
# Em crises, as correlações entre ativos financeiros tendem a subir. Acontece o mesmo com preços? Calculamos, em janelas móveis de 24 meses, a **correlação média entre todos os pares de itens de alimentos** (variação mensal dessazonalizada) e a correlação entre o índice de alimentos e o de energia.

# %%
alim = [i for i in base15 if GRUPO[i] != "Energia"]
XA = TRANSF["Variação mensal dessazonalizada"][alim]
JAN = 24
med_corr = {}
for k in range(JAN, len(XA) + 1):
    w = XA.iloc[k - JAN:k].dropna(axis=1, thresh=JAN - 2)
    c_ = w.corr().values
    med_corr[XA.index[k - 1]] = c_[np.triu_indices_from(c_, 1)].mean()
med_corr = pd.Series(med_corr)

lA = np.log(IDX["Alimentos (todos)"]).diff(); lA = lA - lA.groupby(lA.index.month).transform("mean")
lE = np.log(IDX["Energia"]).diff(); lE = lE - lE.groupby(lE.index.month).transform("mean")
rollAE = lA.rolling(JAN).corr(lE)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    subplot_titles=("Correlação média entre pares de alimentos (janela de 24 meses)",
                                    "Correlação Alimentos × Energia (janela de 24 meses)"))
fig.add_scatter(x=med_corr.index, y=med_corr, line=dict(color=SERIES[0], width=2), showlegend=False, row=1, col=1,
                hovertemplate="Janela até %{x|%m/%Y}: %{y:.3f}<extra></extra>")
fig.add_scatter(x=rollAE.index, y=rollAE, line=dict(color=SERIES[1], width=2), showlegend=False, row=2, col=1,
                hovertemplate="Janela até %{x|%m/%Y}: %{y:.2f}<extra></extra>")
for r_ in (1, 2):
    fig.add_vrect(x0="2020-03-01", x1="2023-06-01", fillcolor=GRID, opacity=0.5, line_width=0, layer="below", row=r_, col=1)
    fig.add_hline(y=0, line_color=AXIS, row=r_, col=1)
fig.update_layout(height=620, title="Correlações em janela móvel; a data é o fim da janela (faixa cinza: pandemia e choque inflacionário, 2020–23)")
fig.show()
md(f"* Correlação média entre alimentos: mediana de {med_corr.median():.3f} nas janelas, máximo de {med_corr.max():.3f} (janela encerrada em {med_corr.idxmax():%m/%Y}). "
   f"Os preços de alimentos passaram a se mover juntos quando a pandemia atingiu a cadeia toda ao mesmo tempo.\n"
   f"* Alimentos × Energia: média de {rollAE[:'2021-12-01'].mean():+.2f} nas janelas até 2021 e {rollAE['2022-06-01':].mean():+.2f} a partir de meados de 2022. "
   f"**A relação mudou de sinal**: antes os choques de energia e de comida eram independentes ou opostos; depois de 2022 passaram a andar juntos.")

# %% [markdown]
# ---
# ## 4. Defasagens: energia antecipa alimentos?
# Diesel move caminhões, tratores e a logística toda; seria natural que o custo de energia chegasse aos alimentos **com atraso**. Testamos de duas formas:
# 1. **Correlação cruzada:** correlação entre a variação do diesel no mês *t* e a variação dos alimentos no mês *t + k*, para *k* = 0 a 12.
# 2. **Teste de causalidade de Granger:** o passado do diesel melhora a previsão dos alimentos além do passado dos próprios alimentos? (Granger, 1969). Repetido para cada grupo de alimento, com correção FDR.
#
# *Granger não prova causalidade econômica; prova apenas precedência estatística útil para previsão.*

# %%
energia = {"Diesel": "Automotive diesel fuel", "Gasolina comum": "Gasoline, unleaded regular", "Eletricidade": "Electricity",
           "Gás encanado": "Utility (piped) gas"}
def ccf(x, y, lags=range(0, 13)):
    """corr(x_t, y_{t+k})"""
    return pd.Series({k: x.corr(y.shift(-k)) for k in lags})

d12 = logP.diff(12)
alim12 = np.log(IDX["Alimentos (todos)"]).diff(12)
cc = pd.DataFrame({nome: ccf(d12[it], alim12) for nome, it in energia.items()})
cc_m = pd.DataFrame({nome: ccf(dlog_ds[it], lA) for nome, it in energia.items()})
n_eff = lA.dropna().shape[0]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Variação em 12 meses", "Variação mensal dessazonalizada"))
for j, nome in enumerate(energia):
    fig.add_scatter(x=cc.index, y=cc[nome], name=nome, mode="lines+markers", line=dict(color=SERIES[j], width=2),
                    marker=dict(size=8), row=1, col=1, legendgroup=nome,
                    hovertemplate=f"{nome}<br>defasagem %{{x}} meses: r = %{{y:.2f}}<extra></extra>")
    fig.add_scatter(x=cc_m.index, y=cc_m[nome], name=nome, mode="lines+markers", line=dict(color=SERIES[j], width=2),
                    marker=dict(size=8), row=1, col=2, legendgroup=nome, showlegend=False,
                    hovertemplate=f"{nome}<br>defasagem %{{x}} meses: r = %{{y:.2f}}<extra></extra>")
banda = 1.96 / np.sqrt(n_eff)
fig.add_hrect(y0=-banda, y1=banda, fillcolor=GRID, opacity=0.6, line_width=0, layer="below", row=1, col=2)
fig.update_xaxes(title_text="Defasagem k (meses que a energia antecede os alimentos)")
fig.update_yaxes(title_text="Correlação", row=1, col=1)
fig.update_layout(title="Correlação cruzada: energia no mês t × índice de alimentos no mês t + k (faixa: ±1,96/√n)",
                  height=520, margin=dict(t=130), legend=dict(orientation="h", y=1.1, yanchor="bottom", x=0))
fig.show()
melhor = cc.idxmax()
md(("Defasagem de correlação máxima (variação em 12 meses): " +
                 "; ".join(f"**{n}**: {melhor[n]} meses (r = {cc.loc[melhor[n], n]:.2f})" for n in energia)))

# %%
def granger_p(y, x, maxlag=6):
    d = pd.concat([y, x], axis=1).dropna()
    with contextlib.redirect_stdout(io.StringIO()):          # compatível com versões novas e antigas do statsmodels
        r = grangercausalitytests(d.values, maxlag=maxlag)
    return {lag: r[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)}

alvos = {g: dI_ds[g] for g in GRUPOS if g != "Energia"}
alvos["Alimentos (todos)"] = lA
lin = []
for nome_e in ["Diesel", "Gasolina comum"]:
    x = dlog_ds[energia[nome_e]]
    for g, y in alvos.items():
        ps = granger_p(y, x)
        best = min(ps, key=ps.get)
        lin.append(dict(causa=nome_e, efeito=g, melhor_defasagem=best, p_min=ps[best], p_lag3=ps[3], p_lag6=ps[6]))
gr = pd.DataFrame(lin)
gr["p_lag6_FDR"] = multipletests(gr["p_lag6"], method="fdr_bh")[1]
gr["conclusão (lag 6, FDR 5%)"] = np.where(gr["p_lag6_FDR"] < 0.05, "precede ✔", "não detectado")
gr.sort_values("p_lag6").style.format({"p_min": "{:.3f}", "p_lag3": "{:.3f}", "p_lag6": "{:.3f}", "p_lag6_FDR": "{:.3f}"}, decimal=",").hide(axis="index")

# %% [markdown]
# *Nota metodológica:* o p-valor mínimo entre seis defasagens (`p_min`) é mostrado só como referência, porque escolher a melhor defasagem depois de olhar os dados infla a significância. A conclusão usa uma defasagem fixada antes (6 meses) e corrige para os 16 testes.

# %% [markdown]
# ### 4.1 Quanto do diesel chega aos alimentos? Regressão com defasagens distribuídas
# Regressão da inflação de alimentos em 12 meses contra a inflação do diesel em 12 meses defasada de 0 a 12 meses, com erros-padrão HAC (Newey-West, 12 lags), que corrigem a autocorrelação criada pela sobreposição das janelas.

# %%
k_opt = int(melhor["Diesel"])
dfr = pd.DataFrame({"alim": alim12 * 100, "diesel": d12["Automotive diesel fuel"].shift(k_opt) * 100,
                    "alim_lag12": alim12.shift(12) * 100}).dropna()
mod = sm.OLS(dfr["alim"], sm.add_constant(dfr[["diesel"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
md((
    f"**Modelo:** inflação 12m de alimentos(t) = α + β · inflação 12m do diesel(t − {k_opt}) + ε  (n = {int(mod.nobs)})\n\n"
    f"* β = **{mod.params['diesel']:.3f}** (IC 95%: {mod.conf_int().loc['diesel', 0]:.3f} a {mod.conf_int().loc['diesel', 1]:.3f}); "
    f"p = {mod.pvalues['diesel']:.4f}; R² = {mod.rsquared:.2f}.\n"
    f"* Leitura: um aumento de 10% no diesel está associado a cerca de **{10 * mod.params['diesel']:.1f} p.p.** a mais de inflação de alimentos "
    f"{k_opt} meses depois. " + ("O coeficiente **é** significativo a 5% mesmo com erros HAC. " if mod.pvalues['diesel'] < 0.05 else
    "Com erros HAC, porém, o coeficiente **não é significativo a 5%**: a correlação alta vem em boa parte de um único episódio (2021–22). ") +
    "Em qualquer caso é associação, não efeito causal isolado: 2021–22 teve também choque de demanda, salários e cadeias de suprimento."))
fig = px.scatter(dfr.reset_index(), x="diesel", y="alim", trendline="ols", color_discrete_sequence=[SERIES[0]],
                 hover_data={"date": "|%m/%Y"},
                 labels={"diesel": f"Diesel: variação 12m em t − {k_opt} (%)", "alim": "Alimentos: variação 12m em t (%)"})
fig.update_traces(marker=dict(size=8, opacity=.75, line=dict(width=1, color=SURFACE)))
fig.update_layout(title=f"Diesel ({k_opt} meses antes) × inflação de alimentos", height=460)
fig.show()

# %% [markdown]
# ---
# ## 5. Gráfico interativo sem servidor: escolha qualquer item
# Este gráfico usa o menu nativo do Plotly e funciona mesmo na versão estática (nbviewer/HTML), sem precisar executar o notebook.

# %%
itens_menu = sorted(largo.columns, key=lambda i: (GRUPO[i], ROT[i]))
k0 = itens_menu.index("Eggs, grade A, large")   # item exibido ao abrir
fig = go.Figure()
for k, it in enumerate(itens_menu):
    fig.add_scatter(x=datas, y=P[it], name="Nominal", visible=(k == k0), line=dict(color=SERIES[0], width=2),
                    hovertemplate="%{x|%m/%Y}: US$ %{y:.3f}<extra>Nominal</extra>")
    fig.add_scatter(x=datas, y=R[it].where(P[it].notna()), name="Real (US$ jul/2026)", visible=(k == k0),
                    line=dict(color=SERIES[1], width=2, dash="dot"),
                    hovertemplate="%{x|%m/%Y}: US$ %{y:.3f}<extra>Real</extra>")
botoes = []
for k, it in enumerate(itens_menu):
    vis = [False] * (2 * len(itens_menu)); vis[2 * k] = vis[2 * k + 1] = True
    botoes.append(dict(label=f"{ROT[it]} ({GRUPO[it]})", method="update",
                       args=[{"visible": vis}, {"title.text": f"{ROT[it]}: preço médio nos EUA ({resumo.set_index('item').loc[it, 'unit']})"}]))
fig.update_layout(updatemenus=[dict(buttons=botoes, active=k0, direction="down", x=0, xanchor="left", y=1.18, yanchor="top",
                                    showactive=True, bgcolor=SURFACE)],
                  title=dict(text=f"{ROT[itens_menu[k0]]}: preço médio nos EUA ({resumo.set_index('item').loc[itens_menu[k0], 'unit']})", y=0.92), height=520,
                  yaxis_title="US$", hovermode="x unified", legend=dict(orientation="h", y=-0.15, x=0),
                  margin=dict(t=130))
fig.show()

# %% [markdown]
# ---
# ## 6. Painel interativo (precisa do notebook em execução)
# Os controles abaixo funcionam no Colab ou no Jupyter. Na versão estática aparece só o resultado com os valores iniciais.

# %% [markdown]
# ### 6.1 Comparador de pares
# Escolha dois itens e a transformação. O painel mostra: o diagrama de dispersão com a reta de regressão, a correlação em janela móvel e a correlação cruzada por defasagem (A antecede B quando *k* > 0).

# %%
opcoes = [(f"{ROT[i]} · {GRUPO[i]}", i) for i in itens_menu]
TRANSF_ALL = {
    "Variação mensal dessazonalizada": dlog_ds,
    "Variação em 12 meses": logP.diff(12),
    "Nível (cuidado: espúria)": Pf,
}

def comparar(item_a, item_b, transformacao, janela):
    M_ = TRANSF_ALL[transformacao]
    d = pd.concat([M_[item_a], M_[item_b]], axis=1, keys=["a", "b"]).dropna()
    if len(d) < 24 or item_a == item_b:
        print("Escolha dois itens diferentes com pelo menos 24 meses em comum."); return
    r, p = stats.pearsonr(d["a"], d["b"]); rho = d.corr(method="spearman").iloc[0, 1]
    esc = 1 if transformacao.startswith("Nível") else 100
    rol = M_[item_a].rolling(janela, min_periods=int(janela * .8)).corr(M_[item_b])
    lags = range(-12, 13)
    cx = pd.Series({k: M_[item_a].corr(M_[item_b].shift(-k)) for k in lags})
    fig = make_subplots(rows=1, cols=3, column_widths=[.34, .38, .28], horizontal_spacing=.08,
                        subplot_titles=(f"Dispersão: r = {r:+.2f} | ρ = {rho:+.2f}", f"Correlação móvel ({janela} meses)",
                                        "Correlação por defasagem"))
    fig.add_scatter(x=d["a"] * esc, y=d["b"] * esc, mode="markers", marker=dict(size=7, color=SERIES[0], opacity=.7),
                    text=d.index.strftime("%m/%Y"), hovertemplate="%{text}<br>A: %{x:.2f}<br>B: %{y:.2f}<extra></extra>",
                    showlegend=False, row=1, col=1)
    b_, a_ = np.polyfit(d["a"] * esc, d["b"] * esc, 1); xs = np.linspace(d["a"].min() * esc, d["a"].max() * esc, 2)
    fig.add_scatter(x=xs, y=a_ + b_ * xs, mode="lines", line=dict(color=SERIES[1], width=2), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=rol.index, y=rol, line=dict(color=SERIES[0], width=2), showlegend=False,
                    hovertemplate="%{x|%m/%Y}: %{y:.2f}<extra></extra>", row=1, col=2)
    fig.add_hline(y=0, line_color=AXIS, row=1, col=2)
    fig.add_bar(x=list(cx.index), y=cx.values, marker_color=[SERIES[1] if k == cx.abs().idxmax() else SERIES[0] for k in cx.index],
                showlegend=False, hovertemplate="k = %{x}: r = %{y:.2f}<extra></extra>", row=1, col=3)
    fig.add_hrect(y0=-1.96 / np.sqrt(len(d)), y1=1.96 / np.sqrt(len(d)), fillcolor=GRID, opacity=.6, line_width=0, layer="below", row=1, col=3)
    unid = "US$" if esc == 1 else "%"
    fig.update_xaxes(title_text=f"A: {ROT[item_a]} ({unid})", row=1, col=1); fig.update_yaxes(title_text=f"B: {ROT[item_b]} ({unid})", row=1, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=2); fig.update_xaxes(title_text="k (meses)", row=1, col=3)
    fig.update_layout(title=f"{ROT[item_a]} × {ROT[item_b]} ({transformacao.lower()})", height=440)
    fig.show()
    print(f"n = {len(d)} meses | Pearson r = {r:+.3f} (p = {p:.2g}) | Spearman ρ = {rho:+.3f} | "
          f"defasagem de maior |r|: {cx.abs().idxmax()} meses (r = {cx[cx.abs().idxmax()]:+.2f})")

w_a = widgets.Dropdown(options=opcoes, value="Automotive diesel fuel", description="Item A", layout=widgets.Layout(width="420px"))
w_b = widgets.Dropdown(options=opcoes, value="Bread, white, pan", description="Item B", layout=widgets.Layout(width="420px"))
w_t = widgets.Dropdown(options=list(TRANSF_ALL), value="Variação em 12 meses", description="Medida", layout=widgets.Layout(width="420px"))
w_j = widgets.IntSlider(value=24, min=12, max=60, step=6, description="Janela")
ui = widgets.VBox([widgets.HBox([w_a, w_b]), widgets.HBox([w_t, w_j])])
display(ui, widgets.interactive_output(comparar, dict(item_a=w_a, item_b=w_b, transformacao=w_t, janela=w_j)))

# %% [markdown]
# ### 6.2 Ranking filtrável de pares

# %%
def ranking(medida, r_min, excluir_familia, so_entre_grupos, grupo, sinal, top):
    d = PARES[medida].copy()
    if excluir_familia: d = d[~d["mesma_familia"]]
    if so_entre_grupos: d = d[~d["mesmo_grupo"]]
    if grupo != "Todos": d = d[(d["grupo_a"] == grupo) | (d["grupo_b"] == grupo)]
    if sinal == "Positivas": d = d[d["r"] > 0]
    elif sinal == "Negativas": d = d[d["r"] < 0]
    d = d[d["r"].abs() >= r_min]
    print(f"{len(d)} pares atendem aos filtros; {int((d['p_fdr'] < .05).sum())} significativos após FDR 5%.")
    if len(d):
        display(mostrar_pares(com_spearman(d.reindex(d["r"].abs().sort_values(ascending=False).index).head(top), TRANSF[medida])))

display(widgets.interactive(
    ranking,
    medida=widgets.Dropdown(options=list(PARES), value="Variação mensal dessazonalizada", description="Medida", layout=widgets.Layout(width="380px")),
    r_min=widgets.FloatSlider(value=0.2, min=0, max=1, step=0.05, description="|r| mínimo"),
    excluir_familia=widgets.Checkbox(value=True, description="Excluir mesma família"),
    so_entre_grupos=widgets.Checkbox(value=False, description="Só entre grupos diferentes"),
    grupo=widgets.Dropdown(options=["Todos"] + GRUPOS, value="Todos", description="Envolvendo"),
    sinal=widgets.ToggleButtons(options=["Ambas", "Positivas", "Negativas"], value="Ambas", description="Sinal"),
    top=widgets.IntSlider(value=15, min=5, max=50, step=5, description="Mostrar")))

# %% [markdown]
# ### 6.3 Cesta personalizada
# Monte uma cesta (uma unidade de cada item escolhido, na unidade do BLS) e veja quanto ela custou ao longo do tempo, em dólares correntes e corrigidos pela inflação. Usa só itens com série completa, para a soma não ter buracos.

# %%
completos = [i for i in itens_menu if Pf[i].notna().all()]
def cesta(itens, real):
    if not itens:
        print("Selecione ao menos um item (Ctrl/Cmd + clique para vários)."); return
    S = (R if real else Pf)[list(itens)]
    tot = S.sum(axis=1)
    fig = go.Figure()
    for k, it in enumerate(itens):
        fig.add_scatter(x=datas, y=S[it], stackgroup="c", name=ROT[it], line=dict(width=0.5, color=SERIES[k % 8]),
                        hovertemplate=f"{ROT[it]}: US$ %{{y:.2f}}<extra></extra>")
    fig.add_scatter(x=datas, y=tot, name="Total", line=dict(color=INK, width=2), hovertemplate="Total: US$ %{y:.2f}<extra></extra>")
    fig.update_layout(title=f"Custo da cesta ({'US$ de jul/2026' if real else 'US$ correntes'})", height=480, hovermode="x unified",
                      yaxis_title="US$")
    fig.show()
    a_, b_ = tot.loc["2015"].mean(), tot.loc[JAN_B[0]:JAN_B[1]].mean()
    print(f"Média 2015: US$ {a_:.2f} | Últimos 12 meses: US$ {b_:.2f} | Variação: {b_ / a_ - 1:+.1%} "
          f"| Pico: US$ {tot.max():.2f} em {tot.idxmax():%m/%Y}")
    if len(itens) > 8: print("Obs.: com mais de 8 itens as cores se repetem; use o hover para identificar.")

padrao = [i for i in ["Eggs, grade A, large", "Milk, fresh, whole, fortified", "Bread, white, pan", "Ground beef, 100% beef",
                      "Coffee, 100%, ground roast, all sizes", "Bananas"] if i in completos]
w_c = widgets.SelectMultiple(options=[(f"{ROT[i]} · {GRUPO[i]}", i) for i in completos], value=padrao, rows=12,
                             description="Itens", layout=widgets.Layout(width="480px"))
w_r = widgets.ToggleButtons(options=[("Nominal", False), ("Real (inflação descontada)", True)], value=False, description="Valores")
display(widgets.HBox([w_c, w_r]), widgets.interactive_output(cesta, dict(itens=w_c, real=w_r)))

# %% [markdown]
# ---
# ## 7. Síntese dos achados
# Os números abaixo são calculados no momento da execução.

# %%
DMx = DM[~DM["mesma_familia"]]
top_entre = DMx[~DMx["mesmo_grupo"]].sort_values("r", ascending=False).iloc[0]
g6 = gr[gr["p_lag6_FDR"] < .05]
maior_real = A.nlargest(3, "var_real"); menor_real = A.nsmallest(3, "var_real")
md((f"""
**Base**
1. A base tem seis problemas que alteram conclusões: café classificado como carne bovina; janelas "de 12 meses" que cobrem até {max(l[2] for l in linhas) if linhas else 12} meses de calendário; bases diferentes (2015 × 2018) misturadas numa mediana única; séries vivas marcadas como descontinuadas; inflação subestimada (38,5% e não 35%); e período-base dos ovos contaminado pelo surto de gripe aviária de 2015.

**Preços**

2. Com a base corrigida ({len(A)} séries), todas subiram em dólares correntes (mediana {A['var_nominal'].median():+.0%}), mas só **{n_real_pos} de {len(A)}** subiram acima da inflação. Mediana real: **{A['var_real'].median():+.1%}**.
3. Maiores altas reais: {', '.join(f"{r.rotulo} ({r.var_real:+.0%})" for r in maior_real.itertuples())}. Maiores quedas reais: {', '.join(f"{r.rotulo} ({r.var_real:+.0%})" for r in menor_real.itertuples())}.
4. A alta real está concentrada no choque de 2020–2022. A energia atingiu o pico de inflação em 12 meses em {pk.set_index('grupo').loc['Energia', 'mes_pico']:%m/%Y}; os alimentos como um todo, em {pk.set_index('grupo').loc['Alimentos (todos)', 'mes_pico']:%m/%Y}.

**Correlações**

5. Em nível, {res.loc['Nível', 'frac_r_maior_05']:.0%} dos pares têm |r| > 0,5. Em variação mensal dessazonalizada, apenas {res.loc['Variação mensal dessazonalizada', 'frac_r_maior_05']:.0%}. **A maior parte da "correlação" entre preços é tendência comum**, não co-movimento.
6. As relações fortes e robustas são dentro da mesma cadeia produtiva (cortes bovinos, combustíveis, laticínios). O par mais forte entre grupos diferentes é **{top_entre['rotulo_a']} × {top_entre['rotulo_b']}** (r = {top_entre['r']:+.2f}). Entre grupos não ligados à carne, as relações de curto prazo são fracas.
7. Parte das correlações entre carne suína e bovina vem do fechamento de frigoríficos na primavera de 2020: sem mar–ago/2020, o par mais forte cai de {top_entre['r']:+.2f} para {TRANSF['Variação mensal dessazonalizada'].loc[SEM_2020, [top_entre['a'], top_entre['b']]].corr().iloc[0, 1]:+.2f}.
8. O PC1 explica só {expl[0]:.0%} da variação mensal: existe um fator comum, mas fraco. A correlação média entre pares de alimentos chega a {med_corr.max():.2f} na janela de 24 meses encerrada em {med_corr.idxmax():%m/%Y} (que cobre a pandemia), contra {med_corr.median():.2f} na mediana das janelas: os preços passam a se mover juntos em crises.

**Defasagens**

9. A inflação do diesel em 12 meses tem correlação máxima com a de alimentos **{melhor['Diesel']} meses depois** (r = {cc.loc[melhor['Diesel'], 'Diesel']:.2f}). No teste de Granger (6 defasagens, FDR 5%), só {len(g6)} de {len(gr)} combinações combustível × grupo passam{(': ' + '; '.join(f"{c} → {e_}" for c, e_ in zip(g6['causa'], g6['efeito']))) if len(g6) else ''}. O diesel, apesar da correlação alta em 12 meses, não passa no teste mensal: a precedência é fraca e concentrada no episódio 2021–22.

**Limitações:** preços nominais médios nacionais (sem regiões, marcas ou tamanhos); CPI-U geral como deflator (não o CPI de alimentos); buracos curtos interpolados; Granger e correlações não identificam causalidade.
"""))

# %% [markdown]
# ## Referências
# * U.S. Bureau of Labor Statistics. *Average Price Data (AP)*. https://www.bls.gov/cpi/factsheets/average-prices.htm, dados em https://download.bls.gov/pub/time.series/ap/
# * U.S. Bureau of Labor Statistics. *CPI for All Urban Consumers (CPI-U), All items, U.S. city average, NSA*, via FRED: série CPIAUCNS. https://fred.stlouisfed.org/series/CPIAUCNS
# * Base Kaggle: *US Grocery and Gas Prices (2015-2026)*, compilação do BLS AP (domínio público, CC0).
# * Benjamini, Y.; Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
# * Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
# * Granger, C. W. J.; Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
# * Newey, W. K.; West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
