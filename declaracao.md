# Declaração de Uso de Inteligência Artificial (IAGen)

Em conformidade com as diretrizes da disciplina para a entrega do trabalho final, a nossa equipe declara o uso pontual de Inteligência Artificial Generativa (ferramenta: **Claude AI**) como suporte durante o desenvolvimento do código. 

Abaixo detalhamos as áreas específicas onde a ferramenta foi utilizada, os prompts empregados e a explicação do funcionamento do código gerado, demonstrando total compreensão da lógica integrada ao projeto.

---

## 1. Cálculo de Métricas (F1-Score, Precisão e Recall)
**Ferramenta:** Claude AI
**Arquivo afetado:** `scripts/eval_metrics.py`

**Prompt empregado:**
> "Como posso implementar uma função em Python usando NumPy para calcular a Matriz de Confusão (Verdadeiros Positivos, Falsos Positivos e Falsos Negativos), a Precisão, o Recall e o F1-Score comparando dois arrays booleanos 8x8 (um com a predição de ocupação do tabuleiro e outro com o ground truth)? Preciso que o código evite o erro de 'divisão por zero'."

**Explicação detalhada do código gerado:**
O trecho gerado e adaptado por nós é responsável por calcular o F1-Score da detecção de ocupação do tabuleiro. O código funciona da seguinte maneira:
1. **Verdadeiros Positivos (TP):** A matriz de predição é comparada com a matriz de ground truth (GT) usando uma operação bitwise `&` ou multiplicação lógica `(pred == 1) & (gt == 1)`. Isso conta as casas que de fato estavam ocupadas e o modelo acertou.
2. **Falsos Positivos (FP):** Identifica as casas onde o modelo detectou uma peça, mas o GT dizia que estava vazia `(pred == 1) & (gt == 0)`.
3. **Falsos Negativos (FN):** Identifica as peças que o modelo não viu `(pred == 0) & (gt == 1)`.
4. **Fórmulas de Cálculo:** 
   - A **Precisão** é calculada como `TP / (TP + FP)`.
   - O **Recall** é calculado como `TP / (TP + FN)`.
   - O código gerado inclui uma estrutura condicional (`if/else`) ou a adição de um termo mínimo (`epsilon = 1e-9`) no denominador. Entendemos que isso é crucial para evitar o `ZeroDivisionError` em casos extremos (por exemplo, se um tabuleiro inteiro for lido como vazio, zerando o denominador da Precisão). 
   - Por fim, o **F1-Score** aplica a média harmônica entre os dois: `2 * (Precisão * Recall) / (Precisão + Recall)`.

---

## 2. Formatação Visual de Gráficos e Bounding Boxes (Matplotlib)
**Ferramenta:** Claude AI
**Arquivo afetado:** Notebooks de visualização e geração de figuras para o relatório.

**Prompt empregado:**
> "Escreva um trecho de código usando Matplotlib que pegue uma imagem de um tabuleiro de xadrez 480x480 retificado, faça um loop para dividi-lo em um grid de 8x8 (60x60 pixels por casa) e desenhe 'bounding boxes' verdes nas casas onde o modelo acertou e vermelhas nas casas onde o modelo errou. Como desenho esses retângulos vazados em cima da imagem original?"

**Explicação detalhada do código gerado:**
Utilizamos a IA para acelerar a geração do código *boilerplate* de visualização, sem focar no processamento da imagem em si, mas apenas na plotagem das figuras que estão no nosso relatório. A lógica compreendida e aplicada é:
1. O código inicializa uma figura e um eixo no matplotlib com `fig, ax = plt.subplots()` e exibe a imagem de fundo (o tabuleiro retificado) com `ax.imshow()`.
2. Um loop iterativo duplo (`for i in range(8): for j in range(8):`) percorre o grid.
3. Para cada iteração, as coordenadas do canto superior esquerdo da casa são calculadas multiplicando os índices pelo tamanho da casa em pixels (ex: `x = j * 60`, `y = i * 60`).
4. A mágica da biblioteca acontece ao instanciar o objeto `matplotlib.patches.Rectangle((x, y), width=60, height=60, edgecolor=cor, facecolor='none', linewidth=2)`. Entendemos que o parâmetro `facecolor='none'` é o que garante que a caixa fique vazada, sem cobrir a peça na imagem.
5. A variável `cor` é preenchida dinamicamente no nosso código: aplicamos a nossa matriz de comparação de resultados, passando 'green' se houve acerto entre predição e gabarito, ou 'red' caso contrário. O patch é então plotado na imagem com `ax.add_patch(patch)`.
