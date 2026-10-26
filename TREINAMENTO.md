# Treinamento do zero: T0 e sprints T1

Este guia parte de um clone sem nenhuma policy treinada. A cadeia é:

```text
T0 (12 DoF, do zero)
├── T1_sprint_arms_locked (12 DoF)
└── T1_sprint_arms_not_locked (21 DoF, pesos expandidos)
    └── eval
```

Os checkpoints oficiais presentes em `T0/` e `T1_sprint_arms_*/iterations/`
são resultados de referência. Os comandos abaixo não dependem deles.

## 1. Instalação

Instale Python, PyTorch, Isaac Gym e as dependências conforme o `README.md` da
raiz. Os treinos requerem uma GPU compatível com CUDA.

O W&B é opcional, mas os yaml históricos vêm com `use_wandb: true`. Se você
não configurou uma conta W&B, troque para `false` antes de começar:

- `envs/T0.yaml`
- `T1_sprint_arms_locked/code/envs/T1_sprint_arms_locked.yaml`
- `T1_sprint_arms_not_locked/code/envs/T1_sprint_arms_not_locked.yaml`

O TensorBoard continua disponível em `logs/<data-hora>/summaries/`.

## 2. Treinar o T0 do zero

Na raiz do repositório:

```sh
python3 train.py --task=T0 --headless True
```

O T0 usa `checkpoint: null`, 12 ações e 10.000 iterações. Ao terminar, anote o
caminho criado:

```text
<RAIZ>/logs/<data-hora>/nn/model_10000.pth
```

Esse arquivo é a entrada dos dois sprints. Substitua `T0_CKPT` abaixo pelo
caminho absoluto real:

```sh
T0_CKPT=/caminho/absoluto/para/logs/<data-hora>/nn/model_10000.pth
```

## 3. Sprint com braços travados

Entre na stack isolada:

```sh
cd T1_sprint_arms_locked/code
```

Execute uma run para cada método:

```sh
python3 train.py --task=T1_sprint_arms_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry none

python3 train.py --task=T1_sprint_arms_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry dup

python3 train.py --task=T1_sprint_arms_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry loss

python3 train.py --task=T1_sprint_arms_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry phase
```

Cada comando gera uma pasta diferente em
`T1_sprint_arms_locked/code/logs/<data-hora>/`. O yaml executa 15.000
iterações. Para usar a mesma convenção da avaliação oficial, use
`nn/model_10000.pth`; o `model_15000.pth` também fica disponível.

Teste uma run em simulação:

```sh
python3 play.py --task=T1_sprint_arms_locked \
  --checkpoint=logs/<data-hora>/nn/model_10000.pth \
  --play_lin_vel_x 2.0 --num_envs=1
```

## 4. Sprint com braços livres

Volte à raiz e entre na segunda stack:

```sh
cd ../../T1_sprint_arms_not_locked/code
```

O runner detecta o checkpoint T0 de 12 DoF e expande a policy para 21 DoF:
mantém os pesos das pernas e inicializa braços e cintura em zero. O termo
`--joint_deviation -1` é obrigatório para reproduzir a iteração oficial
`joint_dev_minus_1`.

`BASE` e `dup` oficiais usam 10.000 iterações:

```sh
python3 train.py --task=T1_sprint_arms_not_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry none --joint_deviation -1 \
  --max_iterations 10000 --seed 42

python3 train.py --task=T1_sprint_arms_not_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry dup --joint_deviation -1 \
  --max_iterations 10000 --seed 42
```

`loss` e `phase` oficiais usam as 15.000 iterações do yaml:

```sh
python3 train.py --task=T1_sprint_arms_not_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry loss --joint_deviation -1 --seed 42

python3 train.py --task=T1_sprint_arms_not_locked --checkpoint="$T0_CKPT" \
  --headless True --symmetry phase --joint_deviation -1 --seed 42
```

Cada run fica em
`T1_sprint_arms_not_locked/code/logs/<data-hora>/`. Teste uma delas com:

```sh
python3 play.py --task=T1_sprint_arms_not_locked \
  --checkpoint=logs/<data-hora>/nn/model_10000.pth \
  --play_lin_vel_x 2.0 --num_envs=1
```

## 5. Usar os treinos novos na avaliação

Abra `eval/policies.yaml` e troque os campos `checkpoint` pelos caminhos das
runs novas, sempre relativos à raiz. Exemplos:

```yaml
checkpoint: T1_sprint_arms_locked/code/logs/<data-hora>/nn/model_10000.pth
```

```yaml
checkpoint: T1_sprint_arms_not_locked/code/logs/<data-hora>/nn/model_10000.pth
```

Para `T0_walk`:

```yaml
checkpoint: logs/<data-hora>/nn/model_10000.pth
```

Não mova somente o `.pth`: a avaliação também lê o `config.yaml` que o
treinador gravou ao lado da pasta `nn/`.

Volte à raiz e siga `eval/README.md`. Comece pelo teste rápido e depois rode a
matriz:

```sh
python3 eval/run_eval.py --policy E1_free_BASE --vcmd 2.0 \
  --num_runs 8 --time_limit 2 --out eval/smoke

python3 eval/run_matrix.py --stage 1
python3 eval/analyze.py
python3 eval/run_matrix.py --stage 2
python3 eval/analyze.py
```
