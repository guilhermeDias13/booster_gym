# Iteração 2026-08-21_v1

## Objetivo

Retreinar a **caminhada omnidirecional** original (juntas superiores fixas, 12 DoF, `T1_locomotion`). Substitui o checkpoint T1 `2026-06-17-02-24-01` que já não está no Vast. Depois serve de base para o T6 (serial 23 DoF).

## Config

Snapshot: [`T0.yaml`](T0.yaml) (cópia de `envs/T0.yaml` = `envs/T1.yaml`).

- Asset: `resources/T1/T1_locomotion.urdf`
- `num_actions: 12`, `num_observations: 47`
- Comandos: `vx, vy, yaw ∈ [-1, 1]`, `gait [1.0, 2.0]`, trimesh
- Scratch (`checkpoint: null`)

## Treino

```sh
python3 train.py --task=T0 --headless True
```

Ckpt final esperado: `logs/<date-time>/nn/model_10000.pth`
