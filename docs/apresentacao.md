---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 22px; }
  h1 { font-size: 36px; color: #1a3a5c; }
  h2 { font-size: 26px; color: #2c5f8a; border-bottom: 2px solid #2c5f8a; }
  img { border-radius: 6px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
  .small { font-size: 18px; }
---

# Analise de Tabuleiros de Xadrez

## Visao computacional classica para leitura de tabuleiros, identificacao e classificacao de pecas

**Davi Ludvig e Julia Macedo**
**Disciplina:** INE410121 / TRV410001 — Visao Computacional - UFSC
**Dataset:** Synthetic Chess Board Images — Kaggle (thefamousrat)

![bg right:45% 90%](images/slide_hero.jpg)

---

## O Problema

**Dada uma imagem de um tabuleiro de xadrez em perspectiva, determinar o estado de cada casa.**

![w:900](images/slide_montage.jpg)

Imagens sinteticas fotorrealistas (Blender Cycles, 1280x1280) com pecas e tabuleiro de **madeira** — material uniforme que cria baixo contraste, desafiando metodos classicos.

---

## Pipeline: Do Pixel ao Mapa de Ocupacao

![w:1100](images/slide_pipeline.jpg)

Cada etapa usa exclusivamente **tecnicas classicas de CV** — sem Deep Learning.

---

## Etapa 1: Pre-processamento

<div class="columns">
<div>

**Dominio do valor:**
- Conversao para tons de cinza
- Suavizacao gaussiana (kernel 5x5)
- Equalizacao de histograma (global e CLAHE)

**Por que:** reduzir ruido antes da deteccao de bordas e normalizar contraste entre imagens com iluminacoes diferentes.

![h:200](images/03b_histogram_equalization.png)

</div>
<div>

![h:200](images/03_preprocessing.png)

**CLAHE** (Contrast-Limited Adaptive Histogram Equalization) faz equalizacao local, preservando detalhes sem saturar regioes uniformes.

</div>
</div>

---

## Etapa 2: Deteccao de Bordas

<div class="columns">
<div>

**4 operadores comparados:**

| Operador | Kernel | Vantagem |
| --- | --- | --- |
| Roberts | 2x2 | Simples, rapido |
| Prewitt | 3x3 | Suavizacao uniforme |
| Sobel | 3x3 | Peso no centro |
| **Canny** | Multi | Bordas finas + histerese |

</div>
<div>

![h:180](images/04b_edge_operators_comparison.png)

**Operacoes morfologicas** refinam o resultado:

![h:160](images/04c_morphological_operations.png)

</div>
</div>

---

## Etapa 3: Hough + Homografia

<div class="columns">
<div>

**Transformada de Hough:**

Cada pixel de borda vota no espaco parametrico (rho, theta). Picos = linhas do grid.

![h:220](images/05_hough_lines.png)

</div>
<div>

**Homografia (4 pontos):**

Correcao de perspectiva — matriz 3x3 que mapeia o tabuleiro angular para visao top-down.

![h:220](images/07_perspective_correction.png)

</div>
</div>

---

## Etapa 4: Segmentacao e Features

<div class="columns">
<div>

![h:250](images/slide_warped_grid.jpg)

Imagem retificada 480x480 dividida em 64 celulas de 60x60 px.

</div>
<div>

**5 features classicas por celula:**

- Intensidade media (grayscale)
- Desvio-padrao (variabilidade)
- Densidade de bordas (Canny)
- Variancia do Laplaciano (textura)
- Diferenca centro-borda (presenca)

**Classificador:** votacao por limiares (>=2 votos = ocupada)

</div>
</div>

---

## Espaco de Features

![w:750](images/09b_feature_space.png)

Scatter plots mostram a separacao entre celulas **ocupadas** (vermelho) e **vazias** (verde). As linhas tracejadas sao os thresholds do classificador.

---

## Resultados: Ocupacao e Cor

<div class="columns">
<div>

**Ocupacao detectada vs Ground Truth:**

![h:220](images/10_occupancy_comparison.png)

</div>
<div>

**Classificacao de cor (HSV Value):**

![h:220](images/10b_piece_color_classification.png)

Pecas claras (boxwood) vs escuras (ebony) diferenciadas pelo canal V do espaco HSV.

</div>
</div>

---

## Deteccao de Jogadas

**Comparacao temporal** de mapas de ocupacao entre dois frames:

![w:700 center](images/12_move_detection.png)

Amarelo = esvaziou | Ciano = ocupou | Cinza = sem mudanca

---

## Roadmap do Projeto

| Etapa | Descricao | Status |
| --- | --- | --- |
| **1. Leitura do tabuleiro** | Detectar, corrigir perspectiva, segmentar 8x8 | Concluido |
| **2. Identificacao de ocupacao** | Quais casas tem pecas + cor (clara/escura) | Concluido |
| **3. Classificacao de pecas** | Tipo: peao, torre, bispo, cavalo, rainha, rei | **Proximo** |
| **4. Indicacao de jogadas** | Notacao algebrica, validacao de lances, PGN | Planejado |

**Atualmente na transicao da Etapa 2 para a Etapa 3.**

---

## Proximos Passos: Classificacao de Pecas

<div class="columns">
<div>

**Abordagens classicas a investigar:**

- **Template matching** (NCC/SSD)
- **Hu Moments** — forma invariante
- **Contornos** — area, perimetro, convexidade
- **HOG** — gradientes orientados

</div>
<div>

**Visao futura (Etapa 4):**

- Reconstruir posicao completa (FEN)
- Gerar notacao algebrica (Nf3, exd5)
- Validar legalidade de lances
- Tracking temporal entre frames

</div>
</div>

---

## Tecnicas Classicas Demonstradas

<div class="columns">
<div>

- Histograma e equalizacao (global + CLAHE)
- 4 operadores de borda (Roberts, Prewitt, Sobel, Canny)
- Operacoes morfologicas (erosao, dilatacao, abertura, fechamento)
- Transformada de Hough
- Homografia e correcao de perspectiva

</div>
<div>

- Features classicas + classificacao por votacao
- Espaco de cores HSV para classificacao
- Analise temporal (deteccao de jogadas)
- Widgets interativos (ipywidgets)
- Compativel com Google Colab

</div>
</div>

---

## Obrigado

### Analise de Tabuleiros de Xadrez

**Repositorio:** [github.com/daviludvig/classical-cv](https://github.com/daviludvig/classical-cv)

**Notebook interativo:** `notebooks/main.ipynb` (funciona no Google Colab)

![bg right:40% 80%](images/slide_warped_grid.jpg)
