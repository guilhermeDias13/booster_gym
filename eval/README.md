# Avaliação das policies

Este diretório avalia nove policies:

- `E1_locked_{BASE,dup,loss,phase}`
- `E1_free_{BASE,dup,loss,phase}`
- `T0_walk`

Os caminhos dos códigos, checkpoints e métodos ficam em `policies.yaml`. Por
padrão, ele aponta para os nove `model_10000.pth` oficiais arquivados neste
repositório. Para avaliar policies recém-treinadas, altere apenas o campo
`checkpoint` de cada entrada. O `config.yaml` da run deve estar duas pastas
acima do checkpoint, ao lado de `nn/`:

```text
logs/<data-hora>/
├── config.yaml
└── nn/
    └── model_10000.pth
```

## Pré-requisitos

Faça a instalação descrita no `README.md` da raiz, incluindo Isaac Gym e
`requirements.txt`. Execute os comandos a partir da raiz do repositório. A
avaliação requer GPU/CUDA; a gravação de vídeo em servidor sem tela requer
`xvfb-run`.

## Teste rápido

Este comando executa oito ambientes durante dois segundos e grava o resultado
em `eval/smoke/`:

```sh
python eval/run_eval.py --policy E1_free_BASE --vcmd 2.0 \
  --num_runs 8 --time_limit 2 --out eval/smoke
```

Para também gravar dois ambientes:

```sh
xvfb-run -a python eval/run_eval.py --policy E1_free_BASE --vcmd 2.0 \
  --num_runs 8 --time_limit 2 --out eval/smoke --record_env 0 3
```

## Matriz completa

A etapa 1 executa as condições iniciais de todas as policies. Em seguida, a
análise cria `eval/analysis/summary.csv` e marca quais são viáveis:

```sh
python eval/run_matrix.py --stage 1
python eval/analyze.py
```

A etapa 2 usa esse `summary.csv` e executa condições adicionais somente para
as policies viáveis:

```sh
python eval/run_matrix.py --stage 2
python eval/analyze.py
```

Use `--only NOME [NOME ...]` para limitar policies. Jobs que já possuem
`runs.npz` são ignorados.

## Vídeos representativos

Depois de `analyze.py`, grave os ambientes mediano, melhor e pior indicados no
`summary.csv`:

```sh
xvfb-run -a python eval/run_matrix.py --videos --vcmd 2.0
```

## Executar em uma máquina remota

No PowerShell, monte e envie o workspace necessário:

```powershell
.\eval\upload.ps1 -HostIp <IP> -Port <PORT> -Label "avaliação"
```

O script envia os códigos de T0 e dos dois sprints, os scripts de avaliação,
os checkpoints e seus `config.yaml`. O workspace remoto fica em
`/app/eval_ws`.

## Saídas

- `eval/results/<policy>/<condição>/runs.npz`: séries de todas as corridas
- `eval/results/<policy>/<condição>/meta.json`: configuração e proveniência
- `eval/results/<policy>/<condição>/videos/*.mp4`: vídeos solicitados
- `eval/analysis/*.csv`: métricas por corrida, resumos e comparações

`python eval/train_logs.py` resume os eventos do TensorBoard encontrados em
`logs/<data-hora>/summaries/`. Esses eventos são produzidos por um treino
local e não fazem parte do resultado oficial compacto arquivado no repositório.
