# Roteiro do vídeo

O vídeo gravado está publicado em [youtu.be/b-KZiR1OuFc](https://youtu.be/b-KZiR1OuFc).
Este roteiro registra o que foi mostrado e serve de base para novas gravações.

O vídeo precisa mostrar o detector funcionando com os pesos treinados neste projeto.
A duração sugerida é de cerca de cinco minutos. Cada bloco pode ficar com um integrante.

## Antes de gravar

1. Rodar `python -m app.main --device 0` (ou `--device cpu`) e abrir `http://127.0.0.1:7860`
   em tela cheia, com zoom entre 110% e 125%.
2. Fazer uma análise de aquecimento: a primeira carrega o modelo e demora mais. Depois,
   recarregar a página para voltar ao estado vazio.
3. Conferir no canto superior direito "Modelo pronto em GPU 0" (ou "em CPU").
4. Gerar o relatório com `python scripts/build_report.py`, para que o botão
   "Ler o relatório técnico" abra a cópia local.
5. Fechar abas e notificações. Para a webcam, deixar uma embalagem limpa e um segundo
   terminal prontos.

## Blocos

### 1. Abertura (0:00 a 0:30)

Tela: aba **Analisar**, vazia.

> "Olá, nós somos [nomes]. Este é o OlhoLimpo, um detector de resíduos em imagens e vídeos.
> Dado um registro de uma calçada, praça ou margem de rua, o modelo marca onde há lixo
> visível, para que uma pessoa revise só essas regiões. O detector foi treinado por nós com
> o conjunto TACO, e tudo roda localmente nesta máquina."

### 2. Dados e treinamento (0:30 a 1:15)

Tela: **Sobre o modelo**, seção **Dados**; depois **Treinamento** no índice lateral.

> "Das 1.500 imagens do TACO, 778 tinham licença registrada e caixas válidas. Juntamos as
> 60 categorias em uma só, resíduo. Separamos 544 imagens para treino, 117 para validação e
> 117 para teste, antes de qualquer aumento de dados."

> "Partimos de um YOLO26n pré-treinado e ajustamos por 60 épocas numa RTX 3060, em cerca de
> 19 minutos. Checkpoint, resolução de 640 pixels e limiar de 0,35 foram escolhidos só com a
> validação; o teste ficou guardado para o fim. Uma rodada de uma época tinha AP50 de 2,4% na
> validação, e o ajuste completo chegou a 47,4%."

### 3. Imagem (1:15 a 2:30)

1. Clicar em **Analisar** e na amostra **Vários resíduos no solo**.
   > "Esta imagem é do teste reservado; o modelo nunca a viu no treino. A confiança mínima
   > já vem em 0,35, o valor escolhido na validação."
2. Clicar em **Analisar imagem**.
   > "O modelo encontrou [N] resíduos. Cada caixa tem número e confiança, e a lista ao lado
   > traz o tempo de processamento e as coordenadas."
3. Passar o mouse sobre os itens da lista.
   > "Passando o mouse na lista, a caixa correspondente fica em destaque."
4. Baixar a confiança para 0,20, analisar de novo e voltar para 0,35.
   > "Com limiar menor aparecem mais candidatos, e também mais falsos positivos. Por isso
   > usamos o valor da validação."
5. Clicar em **JSON** ou **Imagem anotada**.
   > "O resultado pode ser exportado em PNG, JSON e CSV."

### 4. Vídeo (2:30 a 3:00)

Clicar na amostra **Montagem de imagens do teste** e em **Analisar vídeo**. Ao terminar,
dar play e apontar o resumo.

> "O mesmo detector roda quadro a quadro. Este vídeo é uma montagem de imagens do teste,
> não uma gravação de câmera. O resumo separa o FPS do arquivo do FPS de processamento, e
> um mesmo objeto em vários quadros conta várias vezes."

### 5. Webcam, opcional (3:00 a 3:20)

Rodar `olholimpo webcam --device 0`, mostrar uma embalagem limpa, apertar **S** para salvar
um quadro e **Q** para sair.

> "Para tempo real, a webcam roda por este comando. Este é um ensaio controlado, não um
> teste em ambiente urbano."

Se este bloco for gravado, a limitação "a webcam não foi testada com uma câmera física"
precisa ser atualizada no relatório.

### 6. Resultados (3:20 a 4:15)

Tela: **Sobre o modelo**, **Resultado no teste reservado**; depois **Onde erra**, imagens
**000115** e **000104**.

> "No teste, com 117 imagens e 395 objetos, a precisão foi 54,2%, o recall 35,7%, o AP50
> 38,5% e o AP50-95 25%. Nenhuma métrica chegou à nossa meta interna de 0,75. O modelo
> encontrou 141 objetos e perdeu 254: quando aponta algo, acerta pouco mais da metade das
> vezes, mas deixa passar quase dois terços dos objetos."

> "Nesta imagem, três objetos pequenos junto à cerca não foram detectados. Nesta outra,
> com plásticos amontoados, 12 objetos ficaram sem correspondência. Objetos pequenos,
> sobrepostos e com fundo seco concentram as perdas."

### 7. Encerramento (4:15 a 5:00)

Tela: **Reproduzir** e **Limites conhecidos**; mostrar o botão do relatório.

> "Código, pesos e relatório estão no GitHub, e estes comandos reproduzem tudo, do preparo
> dos dados à aplicação. O OlhoLimpo cumpre o ciclo completo de um projeto de visão
> computacional, mas o resultado mostra que ele serve como apoio à triagem, sempre com
> revisão humana. O próximo passo é testar com imagens próprias e em campo. Obrigado."

## Cuidados

- Falar o número que aparecer na tela; ele pode variar entre CPU e GPU.
- Não dizer que o modelo funciona em produção ou que encontra todo o lixo.
- Não acelerar a reprodução nem cortar a medição de tempo sem avisar.
- Usar apenas números que estão em `reports/metrics.json` e no relatório.
