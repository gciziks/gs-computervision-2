# Dados de treinamento e avaliação

## Origem

O projeto usa as anotações oficiais do TACO, publicadas por Pedro F. Proença e Pedro Simões, no commit `3844bcd3535cd12ec0218494d1d79f1372f553cc` do repositório dos autores. A versão contém 1.500 imagens e 4.784 anotações de objetos. As contagens efetivamente utilizadas estão em [summary.json](../data/manifests/summary.json); não são estimativas nem incluem aumentos como novas imagens.

- Repositório: https://github.com/pedropro/TACO
- Artigo: https://arxiv.org/abs/2003.06975
- SHA-256 das anotações: `812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b`.

## Conjunto preparado

Foram obtidas 778 imagens com 2.594 caixas, sem falhas de download. O filtro excluiu 715 imagens sem licença registrada e 7 com anotações inválidas. A preparação normalizou a orientação EXIF de 160 imagens. Não foi necessário recorrer às miniaturas alternativas nesta execução.

| Conjunto | Imagens | Caixas |
| --- | ---: | ---: |
| Treino | 544 | 1.854 |
| Validação | 117 | 345 |
| Teste | 117 | 395 |
| Total | 778 | 2.594 |

Foram encontrados 778 grupos pelo critério de origem/hash adotado, todos com uma imagem. Isso significa que o algoritmo não encontrou duplicatas ou pares próximos no limiar definido, não que toda relação entre cenas foi descartada por inspeção humana.

## Seleção e licenças

O JSON de origem não contém uma tabela de licenças preenchida. Das imagens, 715 têm licença nula, 319 registram apenas `CC` e 466 registram `ODBL (c) OpenLitterMap & Contributors`. O preparo padrão exclui as 715 com licença nula e preserva as strings das demais. `CC` não informa a variante da licença; o registro ODbL também não deve ser interpretado como uma autorização irrestrita para reproduzir qualquer mídia.

Essa seleção é um filtro de metadados, não uma verificação jurídica individual de cada fotografia. Imagens brutas não integram o código versionado. Antes de redistribuir fotografias ou usá-las em publicação externa, conferir os termos aplicáveis e fornecer atribuição. A licença MIT do toolkit não substitui as condições das imagens.

O parâmetro `--include-unrecorded-license` permite uma análise de pesquisa separada das imagens sem metadado de licença. Não foi adotado no preparo padrão e não concede direito de redistribuição.

## Transformações

As categorias originais são agregadas em uma classe, `0: residuo`. A tabela [class_mapping.json](../data/manifests/class_mapping.json) conserva a correspondência de todas as categorias. O objetivo é localizar resíduos, não identificar material, reciclabilidade, risco sanitário ou intenção de descarte.

As caixas COCO `[x,y,largura,altura]` são convertidas para centro e dimensões normalizadas do formato YOLO. Caixas não finitas, vazias ou fora dos limites e categorias inválidas impedem o uso da imagem; o problema fica registrado em [excluded.json](../data/manifests/excluded.json), sem converter uma imagem problemática em negativo.

O download é retomável, tem tentativas limitadas e verifica imagens já armazenadas por hash. A orientação EXIF é normalizada antes de conferir dimensões, calcular o hash perceptual e preparar a imagem: alguns originais têm pixels em paisagem e anotações em retrato. Quando a origem original falha e existe uma miniatura oficial acessível, o preparo tenta essa alternativa e verifica se a proporção é compatível com a anotação. O manifesto registra a URL realmente obtida, a orientação EXIF e as dimensões. Imagens são convertidas para RGB e reduzidas proporcionalmente para no máximo 1.280 pixels no maior lado; não se amplia a resolução de uma miniatura.

O hash da imagem baixada antes da conversão e o hash do JPEG preparado são registrados separadamente. O JPEG preparado elimina metadados de câmera; suas dimensões podem diferir das dimensões originais das anotações, mas as caixas normalizadas preservam a localização relativa.

## Separação e prevenção de vazamento

Duplicatas de bytes preparados são removidas. Imagens da mesma URL e imagens visualmente semelhantes, com distância de Hamming de dHash de até 6 bits, são agrupadas de forma transitiva. Um grupo nunca aparece em mais de um conjunto.

A separação usa seed 42 e proporções-alvo de 70% treino, 15% validação e 15% teste por quantidade de imagens, respeitando os grupos. As proporções finais podem variar. O manifesto [taco.json](../data/manifests/taco.json) registra ID, grupo, conjunto, hash, URL, licença, dimensões e quantidade de objetos.

O TACO não fornece identidades completas de todas as cenas, trajetos ou objetos físicos. dHash detecta algumas repetições, mas não garante independência de cenas em ângulos muito diferentes. Essa limitação permanece na interpretação dos resultados.

O treinamento aplica aumentos apenas ao conjunto de treino. Validação escolhe checkpoint e limiar; o teste reservado não deve orientar ajustes. Vídeos de demonstração e futuras capturas próprias precisam permanecer separados do treino se forem apresentados como exemplos independentes.

## Auditoria e limitações

Foi inspecionado um painel local com os IDs 291, 432, 485, 356, 256, 194, 487, 156, 447, 206, 193 e 453. As imagens mostram resíduos em solo, vegetação, areia e área construída, com diferenças fortes de tamanho, oclusão e fundo. Há objetos muito pequenos e exemplos em contexto controlado; não são apenas cenas urbanas de descarte espontâneo. A inspeção amostral não certifica a qualidade de todas as anotações.

Também foram conferidas as caixas sobre as imagens 228–233 após normalização de EXIF 6. As caixas acompanham os resíduos na orientação anotada, incluindo objetos pequenos e múltiplos resíduos na mesma cena. Um teste automatizado verifica a rotação de pixels e as dimensões antes da conversão de caixas.

O preparo padrão não inclui negativos independentes revisados nem capturas próprias de teste externo. Portanto, a avaliação não mede adequadamente a taxa de falsos alertas em longos períodos de ambiente limpo nem comprova funcionamento em uma cidade específica. Objetos em uso, como garrafas e copos, podem ser detectados mesmo sem representar lixo descartado.

As falhas de obtenção ficam em [download_failures.json](../data/manifests/download_failures.json). Uma nova execução tenta os arquivos ausentes sem baixar novamente os já verificados. Para reprodução exata de uma avaliação, preservar o manifesto e a configuração da execução; se o conjunto obtido mudar, registrar uma nova versão de dados e repetir a avaliação.

## Reprodução

```powershell
python scripts/prepare_data.py --workers 8
python scripts/train.py --config configs/train.yaml
```

O arquivo `data/processed/taco/data.yaml` aponta para listas explícitas `train.txt`, `val.txt` e `test.txt`. O uso dessas listas evita incorporar arquivos estranhos ao conjunto por varredura de diretórios.
