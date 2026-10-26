# CHANGES — T1_sprint_arms_not_locked vs T1_sprint_arms_locked

Cópia da stack locked. O curriculum 1D, os 15k iters, `action_scale` 1.2 e os métodos `none` / `dup` / `loss` / `phase` são os mesmos.

## Robot

| Item | Locked | Esta task |
|------|--------|-----------|
| URDF | `T1_locomotion` | `T1_serial_head_fixed` (cópia byte-igual do T7) |
| DoF | 12 (pernas) | 21: 8 braço + `Waist` + 12 perna |
| Cabeça | fixa (colapsada) | `AAHead_yaw` e `Head_pitch` continuam `fixed` |
| Obs | 47 | 74 (`11 + 3*21`) |
| Ações | 12 | 21 |

Ordem dos DoF: ombros e cotovelos esquerdos, direitos, `Waist`, perna esquerda, perna direita.

## Pesos

[`code/utils/model.py`](code/utils/model.py) expande um checkpoint T0 12 DoF (47 obs, 12 ações) para 21 DoF. `_N_UPPER = 9`: as primeiras 9 linhas de ação e as colunas de obs de braço/cintura começam a zero. As pernas copiam o walk. Um checkpoint 23 DoF (cabeça livre) não passa neste gate.

## Simetria

[`code/utils/symmetry.py`](code/utils/symmetry.py) usa os 21 nomes. `Waist` espelha em si com sinal negativo. Sem os canais extra de torque/momentum do T9.

## YAML

- Pose de braços do deploy T1: `Shoulder_Pitch` 0.2, `Shoulder_Roll` ∓1.35, `Elbow_Yaw` ∓0.5. Pernas iguais ao locked.
- Gains `Waist` 200/5, `Shoulder` 20/0.5, `Elbow` 20/0.5.
- `penalize_contacts_on` inclui `hand` (mãos livres).
- `basic.description` em ASCII, sem `—`, para o log do wandb não falhar com `UnicodeEncodeError`.

## O que fica igual ao locked

`train.py`, `play.py`, `envs/t1.py`, `envs/base_task.py`, `utils/buffer.py`, `utils/recorder.py`, `utils/terrain.py`, `utils/utils.py`, meshes. Conferido por `tests/test_sameness.py`.
