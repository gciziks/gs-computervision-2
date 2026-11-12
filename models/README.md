# Pesos do OlhoLimpo

O checkpoint customizado fica em `models/best.pt`. A aplicação não baixa pesos genéricos
como alternativa quando esse arquivo está ausente.

Depois da avaliação, `best.json` registra resolução e confiança escolhidas na
validação. Compartilhe esse arquivo e `best.sha256` junto com o checkpoint.

Para gerar o checkpoint:

```powershell
python scripts/prepare_data.py
python scripts/train.py --device 0
```

`models/best.sha256` identifica o arquivo produzido. Os parâmetros resolvidos e o ambiente
usado ficam em `reports/training/args.yaml` e `reports/training.json`.

Pesos `.pt` são ignorados pelo Git. Para compartilhar, anexe `best.pt` a uma release do
repositório e preencha `weights` em `deliverables/links.json`. O destinatário deve comparar
o SHA-256 antes de carregar o arquivo. Só abra checkpoints de origem confiável.

O modelo parte de YOLO26n e recebe ajuste supervisionado para a classe `residuo`.
Os termos do software/modelo Ultralytics e os termos das imagens TACO são distintos.
