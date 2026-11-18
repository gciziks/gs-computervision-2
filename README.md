# OlhoLimpo

Detector customizado de resíduos em imagens, vídeos e webcam. Projeto de Applied
Computer Vision com uma classe agregada (`residuo`) e dados anotados do TACO.

A aplicação destaca objetos visualmente semelhantes aos resíduos do treinamento.
Ela não determina intenção de descarte, propriedade ou reciclabilidade. Ausência
de detecção não comprova que o ambiente esteja limpo.

Integrantes: Cassio Valezzi (RM551059), Gabriel Antony Cadima Ciziks (RM98215),
Lucca Sabatini Tambellini (RM98169) e Victor Nuzzi (RM98209).

Entrega: [vídeo de demonstração](https://youtu.be/b-KZiR1OuFc),
[relatório técnico](https://github.com/gciziks/gs-computervision-2/releases/download/v0.2.0/olholimpo-relatorio.pdf)
e [pesos e configuração](https://github.com/gciziks/gs-computervision-2/releases/tag/v0.1.0).
A release v0.2.0 traz a versão revisada do relatório; os pesos continuam na v0.1.0.

O checkpoint foi treinado por 60 épocas. No teste interno com 117 imagens, obteve
precisão de 54,2%, recall de 35,7%, AP50 de 38,5% e AP50-95 de 25,0%, com confiança
0,35 escolhida na validação. As metas internas de acurácia não foram atingidas;
objetos pequenos e fundos carregados continuam sendo limitações. O relatório
inclui contagens, curvas e cinco casos de erro, sem alegar validação operacional.

## Vídeo de demonstração

[![Vídeo de demonstração do OlhoLimpo no YouTube](https://i.ytimg.com/vi/b-KZiR1OuFc/hqdefault.jpg)](https://youtu.be/b-KZiR1OuFc)

Assista em [youtu.be/b-KZiR1OuFc](https://youtu.be/b-KZiR1OuFc). O vídeo apresenta o
problema, os dados e o treinamento, a aplicação analisando imagens e vídeo, os
resultados do teste reservado e os casos de erro.

## Instalação

Use Python 3.11 ou 3.12. O ambiente verificado nesta máquina usa Python 3.12 e
PyTorch com CUDA 13.0. Uma GPU não é necessária para abrir a aplicação; o treino
é consideravelmente mais rápido com GPU.

Com [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --extra dev --extra report
```

Ou com pip:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[dev,report]"
```

Para GPU NVIDIA compatível, instale a distribuição CUDA do PyTorch depois das
dependências. O driver precisa suportar a versão selecionada:

```powershell
uv pip install --python .venv/Scripts/python.exe torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130 --reinstall --no-deps
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

Nos comandos seguintes, `python` deve apontar ao ambiente virtual. Com uv, use
`uv run --no-sync python` após a instalação CUDA para preservar essa distribuição.

## Preparar dados e treinar

```powershell
python scripts/prepare_data.py --workers 8
python scripts/train.py --device 0
python scripts/evaluate.py --device 0
```

Para treinar em CPU, substitua `--device 0` por `--device cpu`. A configuração
inicial está em `configs/train.yaml`; batch e épocas podem ser alterados por CLI.
No Windows, `workers: 0` evita criação excessiva de processos durante o treino.

O preparo baixa as anotações oficiais de um commit fixo e mantém um cache local.
Imagens sem licença informada são excluídas por padrão. Algumas strings de licença
na origem são incompletas: o manifesto registra essa limitação, sem afirmar que os
direitos de cada imagem foram verificados. Consulte [o data card](docs/DATA-CARD.md).
Não redistribua imagens sem verificar seus termos.

As caixas COCO são convertidas para YOLO, com agrupamento por origem/hash perceptual
antes da divisão. Duplicatas exatas são removidas. Aumentos são aplicados somente
ao treino. Falhas de download são registradas; uma nova execução retenta arquivos
ausentes. Não altere o preparo entre treinar e avaliar: isso pode mudar os splits.

O treino gera `models/best.pt` e `models/best.sha256`. A avaliação escolhe o limiar
na validação antes de ler o teste, preserva as predições e gera `reports/metrics.json`.
Os pesos genéricos de origem não substituem o checkpoint customizado na aplicação.

## Usar a aplicação

Para usar o checkpoint da entrega sem treinar novamente, obtenha os três arquivos
`best.pt`, `best.json` e `best.sha256` da release. Com GitHub CLI autenticado:

```powershell
gh release download v0.1.0 --repo gciziks/gs-computervision-2 --pattern "best.*" --dir models
```

A interface e o terminal leem de `best.json` a resolução e o limiar escolhidos na
validação e conferem a identidade do checkpoint. É possível alterar esses valores
na demonstração; isso não altera os números do teste registrado.

```powershell
python -m app.main --device cpu
```

Abra `http://127.0.0.1:7860` no navegador. Para GPU, use `--device 0`. A aplicação
é uma página própria servida por FastAPI, com duas áreas:

- **Analisar**: arraste ou escolha uma imagem ou um vídeo, ajuste a confiança mínima
  e clique em analisar. As caixas aparecem sobre a imagem, com uma lista sincronizada
  de detecções e downloads PNG/JSON/CSV. Vídeos aceitam até 30 segundos e 100 MB; o
  resumo distingue FPS do arquivo e FPS de processamento. Com FFmpeg no PATH, o vídeo
  anotado é convertido para H.264 e reproduzido no navegador. As imagens de
  `data/demo` aparecem como amostras de um clique.
- **Sobre o modelo**: métricas do teste reservado comparadas à meta interna, divisão
  dos dados, treinamento e curvas, velocidade medida, análise de erros, equipe e
  comandos de reprodução, com link para o relatório técnico. Se
  `output/pdf/olholimpo-relatorio.pdf` existir, a aplicação abre essa cópia local,
  que funciona sem internet. Os números vêm dos registros da execução; o controle de
  confiança da análise não recalcula essas métricas.

Os arquivos exportados ficam em `outputs/demo`, em pastas separadas por execução.
A interface usa apenas `127.0.0.1`; não precisa de hospedagem ou autenticação.

Também é possível usar o terminal:

```powershell
olholimpo caminho/imagem.jpg --device cpu
olholimpo caminho/video.mp4 --device 0
olholimpo webcam --device 0
```

Na webcam, `Q` encerra e `S` salva o quadro anotado e suas detecções. A webcam abre
uma janela OpenCV; é preciso ter câmera física disponível e permissão no Windows.
Um objeto detectado em vários quadros não equivale a vários objetos físicos.

Limites de imagem: 20 MB e 40 milhões de pixels. Vídeos são processados antes da
reprodução, preservando a taxa do arquivo. O codec MP4 de saída depende do OpenCV;
a interface pode convertê-lo para compatibilidade com o navegador.

## Relatório

O [relatório técnico (PDF)](https://github.com/gciziks/gs-computervision-2/releases/download/v0.2.0/olholimpo-relatorio.pdf)
reúne introdução, ferramentas, dados, metodologia, resultados, velocidade, análise de
erros, limitações e conclusão. Os números abaixo vêm de `reports/metrics.json`, a mesma
fonte usada pelo PDF e pela aplicação.

| Dados (TACO) | Imagens | Objetos anotados |
| --- | ---: | ---: |
| Treino | 544 | 1.854 |
| Validação | 117 | 345 |
| Teste reservado | 117 | 395 |

Das 1.500 imagens do TACO, 778 tinham licença registrada e caixas válidas. As 60
categorias foram agrupadas na classe `residuo`.

| Treinamento | Valor |
| --- | --- |
| Pesos de origem | YOLO26n pré-treinado |
| Épocas e lote | 60 épocas, lote 16 |
| Entrada | 640 px (960 px testada na validação, sem ganho) |
| GPU e duração | NVIDIA GeForce RTX 3060, cerca de 19 min |
| Limiar de confiança | 0,35, escolhido na validação |

| Teste reservado | Valor |
| --- | ---: |
| Precisão | 54,2% |
| Recall | 35,7% |
| F1 | 43,1% |
| AP50 | 38,5% |
| AP50-95 | 25,0% |
| Acertos / falsos positivos / não encontrados | 141 / 119 / 254 |
| Latência mediana na GPU | 18,7 ms (51,4 imagens/s) |

Nenhuma métrica atingiu a meta interna de 0,75. A latência conta inferência e desenho
das caixas em imagens já decodificadas, sem leitura de arquivo, câmera ou exibição.
Cinco imagens do teste foram revisadas uma a uma: objetos pequenos, sobrepostos e com
fundo seco concentram as perdas. A avaliação é interna ao TACO, sem coleta própria,
negativos independentes ou ensaio com câmera física.

Para gerar o PDF localmente, a partir de `docs/RELATORIO.json` (texto) e dos registros
em `reports/` (números e figuras):

```powershell
python scripts/build_report.py
```

O arquivo sai em `output/pdf/olholimpo-relatorio.pdf`, e a aplicação passa a abrir essa
cópia. Mais detalhes em [MODEL-CARD](docs/MODEL-CARD.md), [DATA-CARD](docs/DATA-CARD.md)
e [pesos e verificação de integridade](models/README.md).

## Licenças

Código sob AGPL-3.0-only, conforme [LICENSE](LICENSE). O uso de Ultralytics e dos pesos
de origem segue os [termos da ferramenta](https://www.ultralytics.com/license).
Imagens e anotações TACO têm termos próprios; a licença do toolkit não cobre
automaticamente as imagens. Fontes: [TACO](https://github.com/pedropro/TACO) e
[artigo dos autores](https://arxiv.org/abs/2003.06975).
