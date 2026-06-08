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

## Visao computacional classica para leitura de tabuleiros, identificacao, classificacao de pecas e sugestao de jogadas.

**Davi Ludvig e Julia Macedo**
**Disciplina:** INE410121 / TRV410001 — Visao Computacional - UFSC
**Dataset:** Synthetic Chess Board Images — Kaggle (thefamousrat)

![bg right:45% 90%](images/slide_hero.jpg)

---

## O Problema

**Dada uma imagem de um tabuleiro de xadrez em perspectiva, determinar o estado de cada casa.**

![w:900](images/slide_montage.jpg)

Imagens com pecas e tabuleiro de **madeira** — material uniforme que cria baixo contraste, desafiando metodos classicos.

---

## Pipeline: Do Pixel ao Mapa de Ocupacao

![w:1100](images/slide_pipeline.jpg)

Cada etapa usa exclusivamente **tecnicas classicas de CV**.

---

## Etapa 1: Pre-processamento

<div class="columns">
<div>

- Converter para tons de cinza
- Reduzir ruido (suavizacao)
- Melhorar contraste local (equalizacao)

**Objetivo:** preparar a imagem para que as bordas do tabuleiro fiquem mais visiveis.

![h:200](images/03_preprocessing.png)

</div>
<div>

![h:300](images/03b_histogram_equalization.png)

</div>
</div>

---

## Etapa 2: Deteccao de Bordas

<div class="columns">
<div>

Testamos **4 detectores de borda** e escolhemos o que melhor destaca as linhas do tabuleiro (Canny).

Depois, usamos operacoes morfologicas para **limpar ruido** e **conectar bordas** quebradas.

</div>
<div>

![h:180](images/04b_edge_operators_comparison.png)

![h:160](images/04c_morphological_operations.png)

</div>
</div>

---

## Etapa 3: Encontrar o Tabuleiro e Corrigir Perspectiva

<div class="columns">
<div>

**Encontrar as linhas:** a partir das bordas, detectamos as linhas retas que formam o grid do tabuleiro.

![h:220](images/05_hough_lines.png)

</div>
<div>

**Corrigir a perspectiva:** com os 4 cantos do tabuleiro, transformamos a foto angular em uma visao de cima.

![h:220](images/07_perspective_correction.png)

</div>
</div>

---

## Etapa 4: Dividir em Casas e Classificar

<div class="columns">
<div>

![h:250](images/slide_warped_grid.jpg)

Com a visao de cima, dividimos o tabuleiro em **64 casas** iguais.

</div>
<div>

Para cada casa, medimos **5 caracteristicas** (brilho, textura, bordas...) e perguntamos: **tem peca ou nao?**

Se a maioria das medidas indica presenca, a casa e marcada como ocupada.

</div>
</div>

<!-- --- -->

<!-- ## Visualizacao das Caracteristicas

Cada ponto e uma casa. **Vermelho** = tem peca, **verde** = vazia. Quanto mais separados os grupos, melhor a caracteristica funciona.

![w:700](images/09b_feature_space.png) -->

---

## Resultados: Ocupacao e Cor

<div class="columns">
<div>

**Ocupacao detectada vs Ground Truth:**

![h:220](images/10_occupancy_comparison.png)

</div>
<div>

**Classificacao de cor:**

![h:220](images/10b_piece_color_classification.png)

Pecas claras vs escuras diferenciadas pelo brilho de cada casa.

</div>
</div>

---

## Deteccao de Jogadas

Comparando o tabuleiro **antes e depois**, identificamos qual peca se moveu:

![w:700 center](images/12_move_detection.png)

Amarelo = saiu de la | Ciano = chegou aqui

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

## Proximos Passos

<div class="columns">
<div>

**Classificacao de pecas:**

Identificar o **tipo** de cada peca (peao, torre, bispo, etc.) usando forma, contorno e tamanho.

</div>
<div>

**Leitura completa do jogo:**

- Montar a posicao completa do tabuleiro
- Gerar a notacao da jogada (ex: Cavalo para f3)
- Acompanhar a partida ao longo do tempo

</div>
</div>