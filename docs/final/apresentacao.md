---
marp: true
theme: default
paginate: true
math: katex
style: |
  section { font-size: 21px; }
  h1 { font-size: 34px; color: #1a3a5c; }
  h2 { font-size: 26px; color: #2c5f8a; border-bottom: 2px solid #2c5f8a; }
  img { border-radius: 6px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
  .columns3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
  .small { font-size: 17px; }
  .tag { background:#2c5f8a; color:#fff; padding:2px 8px; border-radius:4px; font-size:15px; }
  table { font-size: 17px; }
  pre, code { font-size: 13px !important; }
  .tight { line-height: 1.3; }
  .callout { background:#eaf2fa; border-left:6px solid #2c5f8a; padding:10px 14px; border-radius:4px; margin:14px 0; }
  .big { font-size: 24px; }
  .stage { padding:7px 12px; border-radius:5px; margin:5px 0; font-weight:600; text-align:center; }
  .cls { background:#dceaf7; border:1px solid #9cc3e6; }
  .dl  { background:#fbe3cf; border:1px solid #e6b07a; }
  .arrow { text-align:center; color:#444; font-size:18px; font-weight:600; line-height:1.1; margin:4px 0; }
  .todo { background:#fdecea; border-left:6px solid #c0392b; padding:10px 14px; border-radius:4px; margin:8px 0; color:#7a1f14; font-weight:600; }
---

# Análise de Tabuleiros de Xadrez

## Apresentação Final — Leitura Completa de Partidas com Visão Clássica + Deep Learning

**Davi Ludvig e Julia Macedo**
**Disciplina:** INE410121 / TRV410001 — Visão Computacional - UFSC
**Dataset:** Synthetic Chess Board Images — Kaggle (thefamousrat)

![bg right:40% 90%](../intro/images/slide_hero.jpg)

---

## O Problema

**Dada uma foto de um tabuleiro de xadrez em perspectiva, reconstruir o estado completo do jogo:** ocupação, cor, **tipo** de cada peça, a posição em **FEN** e a **jogada** realizada entre dois estados.

![w:1000](../intro/images/slide_montage.jpg)

<div class="small">

Aplicações: transmissão digital de torneios físicos, alimentação de engines de análise, acessibilidade, ensino de CV.

</div>

---

## Por que uma abordagem híbrida?

<div class="columns">
<div class="tight">

**Clássico** (a disciplina): geometria do tabuleiro, ocupação, cor — transparente e auditável em cada etapa.

**Deep Learning** apenas onde o clássico falha: classificação de **tipo** de peça — silhuetas ambíguas, mesmo material do tabuleiro, perspectiva deformando a forma.

</div>
<div class="tight">

<div class="callout">

O resultado final expõe, lado a lado, onde a visão clássica basta e onde ela deixa de ser competitiva — comparando diretamente as duas famílias de técnicas sobre o mesmo problema.

</div>

</div>
</div>

---

## O Dataset

<div class="columns">
<div>

- ~1 900 imagens 1280×1280 de tabuleiros em ângulo
- Tabuleiro e peças de **madeira** — baixo contraste
- Superfícies, iluminação e ângulos variados
- Anotações: posição das peças + cantos do tabuleiro (GT)
- Sem sequências reais de jogo — posições independentes

</div>
<div>

![w:520](../intro/images/01_dataset_samples.png)

</div>
</div>

---

## Pipeline Completa

![w:1150](../intro/images/slide_pipeline.jpg)

<div class="small">

Bordas (Canny) → linhas do grid (Hough) → homografia (vista de cima) → segmentação em 64 casas → votação de características (**ocupação**) + threshold HSV (**cor**) → **ResNet-34** (**tipo**) → **FEN** → **jogada**.

</div>

---

## Pré-processamento e Bordas

<div class="columns">
<div class="tight">

- Tons de cinza (luminância perceptual) + blur gaussiano 5×5
- Comparação de operadores: Roberts, Prewitt, Sobel, **Canny**
- Canny escolhido: `low=50, high=150` — compromisso entre sensibilidade e robustez
- Morfologia (erosão/dilatação/abertura/fechamento) limpa a imagem de bordas

</div>
<div>

![w:520](../intro/images/04b_edge_operators_comparison.png)

</div>
</div>

---

## Linhas de Hough

A partir das bordas, a **Transformada de Hough** encontra as retas da imagem: cada pixel de borda vota em curvas $(\rho,\theta)$; picos do acumulador indicam linhas reais.

Classificadas em **horizontais** (verde) e **verticais** (azul); janela deslizante seleciona as 9 linhas de grade mais igualmente espaçadas por direção.

![w:820](../intro/images/05_hough_lines.png)

---

## Correção de Perspectiva (Homografia)

Com os 4 cantos do tabuleiro, uma **homografia** transforma a foto em ângulo numa **vista de cima** (480×480) — as 64 casas viram um grid regular, fácil de recortar.

![w:700](../intro/images/07_perspective_correction.png)

<div class="small">

Orientação (0°/90°/180°/270°) resolvida por comparação com o GT quando disponível, ou por densidade de bordas na leitura totalmente automática.

</div>

---

## Segmentação: 64 Casas Individuais

Vista de cima dividida em 8×8 — cada casa vira um recorte independente de 60×60 px, matéria-prima da ocupação e do classificador de peças.

![h:440](../intro/images/08_full_grid.png)

---

## Ocupação: Votação de Características

<div class="columns">
<div class="tight">

4 descritores clássicos por casa, combinados por votação (≥2 de 4):

| Característica | Limiar |
|---|---|
| Desvio-padrão de intensidade | 18 |
| Densidade de bordas (Canny) | 0,04 |
| Variância do Laplaciano | 80 |
| Diferença centro-borda | 12 |

</div>
<div>

![w:520](../intro/images/10_occupancy_comparison.png)

</div>
</div>

---

## Cor da Peça — HSV

Canal **V (brilho)** do espaço HSV comparado a um limiar fixo: peças claras (boxwood) têm brilho maior que escuras (ebony).

<div class="callout big">

Acerto geral: **82,5%** (200 imagens, cantos GT)

</div>

![h:360](../andamento/images/11b_color_cases.png)

---

## O Desafio da Classificação de Tipo

Peças em madeira com tons similares ao tabuleiro dificultam abordagens puramente clássicas:

- Template matching: sensível a escala e rotação
- Confusão entre silhuetas parecidas (peão × bispo, torre × dama)
- Câmera em ângulo deforma a silhueta das peças

<div class="callout">

Solução: manter tudo clássico até aqui, e trocar **só esta etapa** por Deep Learning.

</div>

---

## Solução: Transfer Learning com ResNet-34

<div class="tight">

- **ResNet-34** pré-treinada no ImageNet — features genéricas (bordas, texturas, formas) reaproveitadas
- ~21M parâmetros — suficiente para 12 classes, sem "decorar" as ~62 000 células de treino
- Cabeça final substituída: `Linear(512, 12)` — uma classe por tipo × cor de peça

**Treinamento em duas fases:**

1. **Transfer Learning** (10 épocas) — corpo congelado, só a cabeça nova treina
2. **Fine-tuning** (15 épocas) — rede inteira descongelada, LR discriminativo (menor nas camadas iniciais)

</div>

---

## Construção do Dataset de Treino

<div class="columns">
<div class="tight">

1. Tabuleiro retificado com cantos GT
2. Orientação automática por densidade de bordas
3. Cada casa ocupada salva rotulada em `outputs/piece_cells/{label}/`

**Volume:** ~1 944 imagens × ~32 peças ≈ **62 000 células**

</div>
<div class="small">

| Peça | Qtd. aprox. (cada cor) |
|---|---|
| Peão | ~5 200 |
| Torre | ~5 100 |
| Cavalo | ~5 200 |
| Bispo | ~5 100 |
| Dama | ~5 200 |
| Rei | ~5 200 |

Classes aproximadamente balanceadas.

</div>
</div>

---

## Treinamento — Detalhes

<div class="columns">
<div class="tight">

**Fine-tuning (Fase 2)**

| Hiperparâmetro | Valor |
|---|---|
| LR base (cabeça) | 3e-5 |
| Decaimento por camada | ×0,3 |
| Otimizador | AdamW, wd=1e-4 |
| Label smoothing | 0,1 |
| Warmup | 2 épocas |
| Grad clip | 1,0 |

</div>
<div class="tight">

**Data augmentation**

- Espelhamento horizontal e vertical
- Rotação até 15°
- Brilho/contraste/saturação

<div class="small">

224×224, normalização ImageNet, GPU (CUDA), batch 64.

</div>

<div class="callout">

Impacto do fine-tuning: F1 de 53,2% (1 época) → **91,0%** (15 épocas)

</div>

</div>
</div>

---

## Resultados — Ocupação (pipeline clássica)

<div class="columns">
<div>

**60 imagens · cantos detectados automaticamente**

| Métrica | Valor |
|---|---|
| Recall médio | **70,2%** |
| Precisão média | 59,8% |
| F1 médio | 62,7% |
| Acurácia média | 64,5% |

</div>
<div class="tight">

Recall > precisão → falsos positivos (casas vazias marcadas como ocupadas, baixo contraste madeira-madeira).

<div class="callout big">

Com cantos GT: F1 **63% → 86%**

</div>

O gargalo é **achar os 4 cantos**, não a votação de ocupação.

</div>
</div>

---

## Resultados — Tipo de Peça (ResNet-34)

Avaliado com ocupação GT (isola o classificador do resto da pipeline).

<div class="columns">
<div class="tight">

<div class="callout big">

F1 = **91%** (precisão = recall)
<span class="small">1 412 corretas · 140 erradas · 50 imagens</span>

</div>

<div class="small">

| Peça | Preta | Branca |
|---|---|---|
| Cavalo | **96,4%** | **95,9%** |
| Peão | 96,3% | 93,8% |
| Torre | 89,2% | 90,1% |
| Dama | 87,8% | 90,8% |
| Bispo | 88,2% | 87,6% |
| Rei | **84,5%** ← pior | 89,7% |

</div>

</div>
<div>

![h:320](../andamento/images/15_piece_classification_result.png)

<span class="small">*35/35 peças corretas (100%)*</span>

</div>
</div>

---

## FEN Completa — 6 Campos

`board_to_fen` monta a FEN completa:

- **Posicionamento**: lido do `piece_map` (fileira 8→1, maiúsculas = brancas)
- **5 campos restantes** (lado a mover, roque, en passant, meio-lances, jogada): fornecidos e validados — não inferíveis de 1 frame

`castling_rights_from_position`: estimativa de direitos de roque pela posição de reis e torres.

<div class="small">

```
Início: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

</div>

---

## Ambiguidade Inerente de 180°

Leitura 100% automática (sem GT): orientação por densidade de bordas.

- **0% de erro de eixo** (60 imagens) — o tabuleiro nunca fica de lado
- **~42%** fica invertido em 180° — cores do tabuleiro são **simétricas** sob essa rotação

<div class="callout">

Não é bug, é limite de imagem única. `rotate_piece_map_180` gera a segunda hipótese; o sistema reporta **as duas FENs candidatas** — o tabuleiro real é uma delas.

</div>

---

## Detecção de Jogadas

`detect_moves` compara dois mapas de ocupação: 1 casa esvaziada + 1 ocupada = lance simples.

`classify_move` compara dois `piece_map`s e infere o lance em **UCI** + descrição:

<div class="columns">
<div class="tight">

- Lance simples (`e4`)
- Captura (`exd5`)
- Roque (`O-O`)

</div>
<div class="tight">

- *En passant*
- Promoção (`e8=Q`)

</div>
</div>

<div class="small">

Validado em posições controladas (dataset não tem sequências reais de jogo).

</div>

![h:280](../intro/images/12_move_detection.png)

---

## Pipeline Completa: do Pixel à Jogada

<div class="columns">
<div class="tight">

<div class="stage cls">Imagem original · 1280×1280</div>
<div class="arrow">▼ &nbsp;Hough + homografia</div>
<div class="stage cls">Tabuleiro retificado · 480×480</div>
<div class="arrow">▼ &nbsp;votação de características</div>
<div class="stage cls">Mapa de ocupação 8×8</div>
<div class="arrow">▼ &nbsp;ResNet-34</div>
<div class="stage dl">Mapa de peças · {A1: torre preta, …}</div>
<div class="arrow">▼ &nbsp;board_to_fen</div>
<div class="stage cls">FEN completa (6 campos)</div>
<div class="arrow">▼ &nbsp;classify_move</div>
<div class="stage cls">Notação da jogada (UCI)</div>

</div>
<div class="tight">

| Etapa | Resultado |
|---|---|
| Ocupação | F1 63% · **86%** c/ cantos GT |
| Cor (HSV) | ~82,5% |
| **Tipo da peça** | **F1 91%** |
| FEN + jogadas | Validado (casos controlados) |

<div class="small">

Deep Learning é hoje a etapa **mais precisa**. O gargalo de ponta a ponta é a detecção clássica dos **cantos** do tabuleiro.

</div>

</div>
</div>

---

## Limitações Conhecidas

| Limitação | Impacto | Causa |
|---|---|---|
| Material uniforme (madeira/madeira) | F1 ocupação ≈ 63% ponta a ponta | Baixo contraste peça-fundo |
| Projeção 3D de peças altas | Falsos positivos em casas vizinhas | Torre/dama "vazam" para células adjacentes |
| Ambiguidade de 180° | ~42% invertido (0% erro de eixo) | Simetria de cor do tabuleiro — inerente |
| Limiares fixos | Generalização fraca | Calibrados para este dataset |
| *Domain shift* da CNN | Possível queda em fotos reais | Treinada só com imagens do dataset |

---

## Trabalhos Futuros

<div class="columns">
<div class="tight">

**1 · Reforçar a pipeline clássica**
- Detecção de cantos mais robusta (maior ganho esperado)
- Reduzir falsos positivos de ocupação (limiares adaptativos)

**2 · Validação em jogo real**
- Fotos de tabuleiro físico real
- Sequências reais para validar `classify_move`

</div>
<div class="tight">

**3 · Distribuir o modelo**
- Publicar pesos `.pth` (GitHub Releases / Hugging Face)
- `setup.py` baixando o modelo automaticamente

</div>
</div>

---

## Conclusão

| Componente | Abordagem | Resultado |
|---|---|---|
| Detecção do tabuleiro | Hough + homografia | Robusto; gargalo do sistema |
| Ocupação | Votação de características (clássico) | F1 = 63% · 86% c/ cantos GT |
| Cor da peça | Limiar HSV (clássico) | ~82,5% |
| **Tipo da peça** | **ResNet-34 — transfer learning + fine-tuning** | **F1 = 91%** |
| FEN + jogadas | Regras determinísticas | Validado |

<div class="callout">

Visão clássica (geometria, transparente) + Deep Learning (só onde o clássico falha) — cada abordagem usada onde é mais forte, cobrindo o problema de ponta a ponta: da foto à jogada em notação.

</div>

---

## Links do Projeto

- **Dataset:** [Synthetic Chess Board Images (Kaggle)](https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images)
- **Código-fonte:** [github.com/daviludvig/classical-cv](https://github.com/daviludvig/classical-cv)
- **Checkpoint do modelo (ResNet-34 treinada, .pth):** [GitHub Releases](https://github.com/daviludvig/classical-cv/releases/download/model-v1/piece_classifier_resnet34.pth)
- **Esta apresentação (Google Slides):** [link](https://docs.google.com/presentation/d/1CFl0odjagWv4kNs1oVpdjTxdWwoMgrEh2xZTJH0E9aQ/edit?usp=sharing)
- **Poster (PDF, A1):** [Google Slides](https://docs.google.com/presentation/d/1BDC4UMWAOrh_ubCHu8p4FAyEmWNv6ySB8F8LHkhRvtw/edit?usp=sharing) · [PDF](https://github.com/daviludvig/classical-cv/blob/main/docs/final/poster.pdf)
- **Vídeo (YouTube):** [youtu.be/A4fC6O2yRWQ](https://youtu.be/A4fC6O2yRWQ)

**Obrigado!**
