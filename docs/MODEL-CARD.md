# Modelo OlhoLimpo

## Uso previsto

Localizar objetos visualmente semelhantes aos resíduos anotados no TACO, em imagens, vídeos e captura local de câmera. O modelo destina-se à demonstração acadêmica de detecção de objetos. Não determina se o descarte é irregular, a propriedade do objeto, sua reciclabilidade ou o risco sanitário.

## Identificação e resultados

A fonte dos dados de execução é `reports/metrics.json`. O relatório técnico apresenta os valores efetivamente registrados. O hash do checkpoint identifica os pesos avaliados; uma nova execução requer atualizar as métricas e regenerar o documento.

Os parâmetros completos de treinamento e a origem dos pesos devem acompanhar o checkpoint. O modelo usa uma classe agregada, `residuo`; os rótulos originais são preservados no manifesto do dataset.

A execução registrada concluiu 60 épocas com entrada 640, lote 16 e seed 42, em uma NVIDIA GeForce RTX 3060. O teste interno contém 117 imagens e 395 instâncias: precisão 54,2%, recall 35,7%, AP50 0,385 e AP50–95 0,250. No limiar de confiança 0,35 e IoU 0,50, houve 141 verdadeiros positivos, 119 falsos positivos e 254 falsos negativos. Precisão, recall e AP50 ficaram abaixo das metas internas de 0,75; não há alegação de acurácia operacional.

A validação comparou entradas 640 e 960. A entrada maior não melhorou a AP50–95 segundo a regra registrada em `reports/resolution-selection.json`, e foi mantida a resolução 640. O teste reservado não orientou essa escolha.

O benchmark de 51,38 imagens/s inclui inferência e anotação na GPU sobre imagens já decodificadas. Não inclui captura, leitura de arquivo ou exibição, portanto não é throughput de webcam. A aplicação foi testada em CPU e GPU; a câmera física não foi ativada nesta execução.

## Avaliação

Separar treino, validação e teste antes de aplicar aumentos. Selecionar checkpoint e limiar pela validação e avaliar o teste reservado depois disso. Registrar precisão, recall, F1, AP50 e AP50–95, além de latência e FPS no dispositivo realmente utilizado.

A independência verificada se limita a arquivos distintos e grupos encontrados por dHash. Os 778 grupos resultantes possuem uma imagem cada; metadados incompletos não permitem comprovar independência de cenas, sessões ou objetos físicos. Imagens de um mesmo lugar em ângulos diferentes podem permanecer em conjuntos diferentes. Consultar `docs/DATA-CARD.md`.

Não comparar métricas da classe agregada com métricas de classificadores de materiais ou de pesos genéricos avaliados em COCO. Testes em cenas controladas com embalagens não substituem validação em operação urbana.

## Limitações

- Objetos pequenos, sobrepostos ou em fundos carregados podem não ser detectados.
- Iluminação, desfoque e distância diferentes do treino alteram o resultado.
- Objetos em uso podem ser visualmente iguais a resíduos descartados.
- Cada caixa é uma detecção; sem rastreamento validado, quadros sucessivos não representam objetos únicos.
- A confiança não foi apresentada como probabilidade calibrada de acerto.
- Ausência de detecção não prova ausência de resíduos.

## Dados e licença

Dados: [TACO](https://github.com/pedropro/TACO). Os metadados de licença registrados estão no manifesto e apresentam lacunas; conferir os termos de cada imagem antes de redistribuir. Preservar atribuições. A licença do toolkit não substitui a das imagens.

Ferramenta de treinamento: [Ultralytics](https://www.ultralytics.com/license). A aplicação, os pesos e a redistribuição devem respeitar os termos aplicáveis. Não incluir imagens privadas ou identificáveis de terceiros nas evidências sem autorização.
