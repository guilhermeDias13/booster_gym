# Iteração 2026-09-20_v1

## Objetivo

Uma passagem a partir da caminhada **T0** até corrida reta **> 2 m/s**, no `T1_locomotion` (braços fixos). Curriculum 1D (`curr_v_max`) em vez da escada v4→v8.

## Checkpoint de partida

T0 run `2026-08-21-14-44-42`:

`/app/logs/2026-08-21-14-44-42/nn/model_10000.pth`

## Config

Snapshot: [`T1_sprint_arms_locked.yaml`](T1_sprint_arms_locked.yaml).

| Item | Valor |
|------|-------|
| `max_iterations` | 15000 |
| `min_lin_vel_x` / `lin_vel_start` / `lin_vel_x` | 0.5 / 0.9 / [0.5, 3.5] |
| `gait_frequency` | [1.5, 3.5] |
| `action_scale` | 1.2 |
| `action_rate` / `power` | -0.7 / -1e-3 |
| `feet_air` | não |
| `lin_vel_x_toler` | 0.35 |
| `algorithm.symmetry.method` | **none** (default) |

`dup` / `loss` / `phase` são a **mesma** passagem com `--symmetry`. Não misturar métodos.

## Treino

```sh
cd T1_sprint_arms_locked/code
python3 train.py --task=T1_sprint_arms_locked --symmetry none \
  --checkpoint=/app/logs/2026-08-21-14-44-42/nn/model_10000.pth \
  --headless True
# repetir com --symmetry dup | loss | phase
```

Archive por método: `BASE/` (`none`), `dup/`, `loss/`, `phase/` — cada um com `videos/` e `logs/<date-time>/`.

| Método | Run | Vídeo |
|--------|-----|-------|
| BASE | `2026-09-20-21-01-54` | `BASE/videos/2026-09-21-11-56-39.mp4` |
| dup | `2026-09-21-12-02-13` | `dup/videos/2026-09-22-02-54-13.mp4` |
| loss | `2026-09-22-03-11-16` | `loss/videos/2026-09-22-11-23-23.mp4` |
| phase | `2026-10-20-05-26-56` | a correr, epoch 14/15000 |

A run `2026-09-22-11-35-24` não ficou no repo. O `phase` foi relançado em 2026-10-20 02:27 (UTC-3) na Máquina 4 — `220.130.209.122:38540` (`https://220.130.209.122:38572/terminals/1`, hostname `C.56577784`). W&B offline: `logs/2026-10-20-05-26-56/wandb/offline-run-20261020_052656-kgqoxujy`. Checkpoint `/app/checkpoints/T0_model_10000.pth`. O log diz `symmetry method: phase`.

`Skipping curriculum load: checkpoint shape (21, 21) != env shape (25, 25)` é esperado: a grade 2D do T0 não cabe na deste YAML, e o currículo 1D não usa essa grade. O `UnicodeEncodeError` é só o log do wandb ao imprimir o travessão da `description`; o treino seguiu.

## Eval

```sh
python3 play.py --task=T1_sprint_arms_locked \
  --checkpoint=logs/<date-time>/nn/model_15000.pth \
  --play_lin_vel_x 2.0 --num_envs=1
```

## Go / no-go

- **Go:** `vel_vx > 2.0` estável, linha reta, sem cair. `curriculum/max_lin_vel_level` > 2.0.
- **Cai:** `action_scale` 1.1 ou `lin_vel_start` 0.7.
- **Teto parado:** `lin_vel_x_toler` 0.40. Não ligar `feet_air`.
