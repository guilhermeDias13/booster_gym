# Joint deviation -1

Mesma passagem que [`joint_dev`](../joint_dev/notes.md), com o termo a `--joint_deviation -1` nos quatro métodos. `loss` e `phase` já tinham corrido a `-1` e foram movidos de `iterations/joint_dev/`. `BASE` (`none`) e `dup` ainda vão ser treinados: as runs a `-0.5` ficam em `iterations/joint_dev/`.

Erro quadrático contra `default_joint_angles`. Pesos por grupo: pitch das pernas 0.5, hip yaw/roll 0.1, resto (braços, cintura, ankle roll) 0.2.

Checkpoint T0: `/app/checkpoints/T0_model_10000.pth`. O arranque expande o 12 DoF: pernas copiadas, braços e cintura a zero. Seed 42. O `BASE` e o `dup` desta pasta correm com `--max_iterations 10000`. `loss` e `phase` já feitos ficaram nas 15000 do YAML.

| Método | Pasta | Estado | Run |
|--------|-------|--------|-----|
| sem simetria | `BASE/videos`, `BASE/logs` | acabou 2026-10-20, `epoch: 10000/10000`, `nn/model_10000.pth`. W&B `offline-run-20261020_130018-g7twzlw5`. Vídeo `BASE/videos/2026-10-20-17-35-05.mp4`. Logs em `BASE/logs/` (`config.yaml` com `joint_deviation: -1.0`, `nn/`, `summaries/`, `wandb/`). A run de `66.114.157.224:41620` (`logs/2026-10-20-06-24-50`, `model_3200.pth`) fica incompleta | Máquina 4 — `220.130.209.122:38540` — `C.56577784` — `https://220.130.209.122:38572/terminals/1` |
| dup | `dup/videos`, `dup/logs` | acabou 2026-10-20, `epoch: 10000/10000`, `logs/2026-10-20-13-14-36/nn/model_10000.pth`. Vídeo `dup/videos/2026-10-20-20-11-09.mp4`. Logs em `dup/logs/` (`config.yaml` com `joint_deviation: -1.0`, `nn/` até `model_10000.pth`, `summaries/`). Sem W&B: o `config.yaml` tem `use_wandb: false` e `/app/logs/2026-10-20-13-14-36/wandb` não existe. Pastas anteriores na mesma máquina: `2026-10-20-13-09-31` e `2026-10-20-13-06-46` | `175.155.64.245:19959` — `C.56893100` — `https://175.155.64.245:19351/terminals/1` |
| loss | `loss/videos`, `loss/logs` | feito (`model_15000.pth`) | `2026-09-26-13-55-52` (`loss/videos/2026-09-27-04-51-31.mp4`) |
| phase | `phase/videos`, `phase/logs` | feito (`model_15000.pth`) | `2026-09-27-05-16-51` (`phase/videos/2026-09-27-13-55-20.mp4`) |

Máquina das runs feitas: `38.117.87.40:43164`.

## Lançamento 2026-10-20 — BASE (`none`)

Máquina nova — `66.114.157.224:41620`, hostname `C.56843592`. Arrancou no terminal web, sem `nohup`. Run `logs/2026-10-20-06-24-50`. W&B offline `offline-run-20261020_062450-l2g1uiyu`. No arranque: `Expanded T1 12-DoF locomotion checkpoint into 21-DoF head-fixed policy (legs copied, arms and waist zeroed)`, `symmetry method: none`. A máquina reiniciou às 08:22:11 (`uptime -s`), quatro minutos depois do último save `nn/model_3200.pth` (08:18). Não há `output.log`. Relançado na Máquina 4 — `220.130.209.122:38540`, hostname `C.56577784`, terminal `https://220.130.209.122:38572/terminals/1`. Run `logs/2026-10-20-13-00-18`. W&B offline `offline-run-20261020_130018-g7twzlw5`. No arranque: `Expanded T1 12-DoF locomotion checkpoint into 21-DoF head-fixed policy (legs copied, arms and waist zeroed)`, `symmetry method: none`, `epoch: 2/10000`. O `dup` a `-1` vai para `175.155.64.245:19959`. Pacote local: `deploy_jd1_overlay.tar`.

## A treinar — BASE (`none`)

```sh
python3 train.py --task=T1_sprint_arms_not_locked \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True --symmetry none --joint_deviation -1 \
  --max_iterations 10000 --seed 42
```

Destino: `BASE/videos/` e `BASE/logs/`.

## A treinar — dup

```sh
python3 train.py --task=T1_sprint_arms_not_locked \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True --symmetry dup --joint_deviation -1 \
  --max_iterations 10000 --seed 42
```

Destino: `dup/videos/` e `dup/logs/`.

## `2026-09-26-13-55-52` — loss

Começou em 2026-09-26. Seed 42. Escala `-1`. Wandb offline: `loss/logs/wandb/offline-run-20260926_135557-4kv356ye`.

```sh
python3 train.py --task=T1_sprint_arms_not_locked \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True --symmetry loss --joint_deviation -1
```

Log de arranque: `Expanded T1 12-DoF locomotion checkpoint into 21-DoF head-fixed policy (legs copied, arms and waist zeroed)`. `T1_sprint_arms_not_locked symmetry method: loss`.

Vídeo: `loss/videos/2026-09-27-04-51-31.mp4`. Checkpoint final: `loss/logs/nn/model_15000.pth`.

## `2026-09-27-05-16-51` — phase

Começou em 2026-09-27. Seed 42. Escala `-1`. Wandb offline: `phase/logs/wandb/offline-run-20260927_051715-b3marbfz`.

```sh
python3 train.py --task=T1_sprint_arms_not_locked \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True --symmetry phase --joint_deviation -1
```

Log de arranque: `Expanded T1 12-DoF locomotion checkpoint into 21-DoF head-fixed policy (legs copied, arms and waist zeroed)`. `T1_sprint_arms_not_locked symmetry method: phase`.

Vídeo: `phase/videos/2026-09-27-13-55-20.mp4`. Checkpoint final: `phase/logs/nn/model_15000.pth`.

## Salvamento

Todo o treino com `--joint_deviation -1` desta task fica **só** dentro de `iterations/joint_dev_minus_1/`, nunca em `iterations/joint_dev/` nem em `iterations/sem_joint_dev/`.

1. Baixar sempre o vídeo **e** a pasta inteira da run (`/app/logs/<date-time>/`: `nn/`, `config.yaml`, `summaries/`, `wandb/`). Se os logs falharem, registar aqui o path remoto que faltou.
2. Destino: `iterations/joint_dev_minus_1/<BASE|dup|loss|phase>/videos/` e `.../logs/`. O mp4 e o conteúdo da run ficam direto nessas pastas, sem a pasta `logs/<date-time>/` no meio.
3. Antes de deixar o material no sítio final, confirmar no `config.yaml` baixado que `joint_deviation: -1.0`.
4. Atualizar a tabela deste ficheiro: máquina com IP e porta, run id (`logs/<date-time>/`), W&B offline, nome do mp4 e checkpoint final.

```powershell
# Máquina N — <HOST>:<PORT> — <metodo> — logs/<date-time>
scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new `
  root@<HOST>:/app/videos/<mp4> `
  .\T1_sprint_arms_not_locked\iterations\joint_dev_minus_1\<BASE|dup|loss|phase>\videos\

scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new -r `
  root@<HOST>:/app/logs/<date-time>/. `
  .\T1_sprint_arms_not_locked\iterations\joint_dev_minus_1\<BASE|dup|loss|phase>\logs\
```
