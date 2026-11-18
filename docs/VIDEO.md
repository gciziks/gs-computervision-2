# Demonstração do OlhoLimpo

O vídeo de CV precisa mostrar o detector funcionando com os pesos treinados neste projeto. A duração de quatro a cinco minutos abaixo é uma sugestão de edição; a página de CV do enunciado não estabelece um limite.

## Antes de gravar

- Preencher os integrantes em `deliverables/team.json` e gerar novamente o relatório.
- Usar o checkpoint e o limiar identificados em `reports/metrics.json`.
- Abrir a aplicação e conferir uma imagem de demonstração independente do treino.
- Separar um vídeo curto e, para a webcam, embalagens limpas em cenário controlado.
- Conferir áudio e tamanho dos textos. Fechar telas com informações pessoais.
- Guardar uma captura da configuração, das curvas e das métricas que serão citadas.

## Roteiro

As partes explicativas podem ser gravadas na aba **Apresentação** da aplicação:
use **Projeto**, **Treinamento**, **Resultados**, **Análise de erros** e
**Reprodução**, nessa ordem. Alterne para **Imagem** e **Vídeo** durante a execução
real. O caso 000115 em Análise de erros inclui a fotografia anotada de uma falha.
As curvas exibidas são de treino/validação; a tabela de Resultados é do teste.

| Tempo sugerido | Mostrar | Explicar |
| --- | --- | --- |
| 0:00–0:30 | Título e aplicação | O problema é localizar resíduos visíveis em registros de espaços públicos. Identificar os integrantes. |
| 0:30–1:15 | Manifesto, configuração e curva de treino | O TACO fornece imagens e anotações. As categorias foram agregadas em uma classe e o modelo foi ajustado com dados separados para treino, validação e teste. Usar as contagens reais da execução. |
| 1:15–2:00 | Inferência em imagem | Executar uma imagem que não participou do treino. Apontar caixas, confiança e tempo. Alterar o limiar e explicar a diferença; o limiar formal está congelado no relatório. |
| 2:00–3:15 | Vídeo e webcam | Demonstrar o processamento contínuo, mantendo visíveis FPS e dispositivo. Variar quantidade, distância e iluminação das embalagens. Identificar a montagem como ensaio controlado. |
| 3:15–4:15 | Métricas e falha real | Mostrar precisão, recall e AP. Apontar uma detecção errada ou objeto não detectado, sem esconder o resultado. Exportar imagem e detecções. |
| 4:15–5:00 | README e arquivos | Mostrar como instalar, obter pesos e executar. Concluir com o que o teste comprovou e as limitações restantes. |

## Cuidados na apresentação

Confiança não é garantia de acerto. Uma garrafa em uso pode ter a mesma aparência de uma descartada; o modelo não conhece a intenção do usuário. Ausência de caixa não prova que a área esteja limpa. Caixas em quadros sucessivos não são uma contagem de resíduos físicos únicos.

Não acelerar a reprodução para sugerir desempenho superior ao medido. Se houver corte para encurtar a gravação, manter a medição de tempo íntegra e explicar a edição. Usar apenas números existentes nos artefatos da execução. Não alegar coleta em campo quando as cenas foram montadas.

## Conferência final

- A gravação inclui execução real, não apenas telas ou slides.
- O checkpoint mostrado é o mesmo do relatório e da demonstração.
- Texto, áudio e caixas são legíveis.
- O vídeo não expõe dados pessoais ou arquivos privados.
- O link foi testado em janela anônima e incluído na entrega.

O arquivo de configuração da entrega é `deliverables/links.json`. Seu campo `video` só deve ser preenchido depois que a gravação real estiver publicada.
