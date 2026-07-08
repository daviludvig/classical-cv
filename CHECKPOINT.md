# Checkpoint da Rede Neural

Este projeto usa uma **ResNet-34** (transfer learning + fine-tuning) para classificar o
tipo/cor das peças de xadrez. Os pesos treinados **não estão versionados neste
repositório** (binário, ~85 MB) — foram publicados separadamente via GitHub Releases.

## Onde está a rede treinada

> **Download:** [`piece_classifier_resnet34.pth`](https://github.com/daviludvig/classical-cv/releases/download/model-v1/piece_classifier_resnet34.pth)
> — GitHub Releases, tag [`model-v1`](https://github.com/daviludvig/classical-cv/releases/tag/model-v1)

- **Arquitetura:** ResNet-34 pré-treinada no ImageNet, cabeça final substituída por `Linear(512, 12)` (uma classe por tipo × cor de peça)
- **Treinamento:** 10 épocas de transfer learning (corpo congelado) + 15 épocas de fine-tuning (rede inteira, LR discriminativo) — ver `src/piece_classifier.py`
- **Resultado:** F1 = 91% na classificação de tipo de peça (avaliado com ocupação ground-truth, 50 imagens)

## Como usar

1. Baixe o arquivo do link acima.
2. Coloque-o em `models/piece_classifier_resnet34.pth`.
3. Rode as células de inferência em `notebooks/main.ipynb`.

Para retreinar do zero, delete o arquivo e rode a célula de treinamento do notebook —
ela reconstrói o dataset de células rotuladas e treina o modelo nas duas fases descritas
acima.

Detalhes completos de hiperparâmetros, data augmentation e métricas estão no
[relatório final](docs/final/relatorio/main.pdf), Seção 4 (Deep Learning) e Seção 6
(Resultados).
