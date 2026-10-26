# Como modificar T1_sprint_arms_locked

Primeiro o YAML. Só mexer no `.py` se o knob não existir.

Ficheiros que podes editar:

- [`code/envs/T1_sprint_arms_locked.yaml`](code/envs/T1_sprint_arms_locked.yaml)
- [`code/envs/t1_sprint_arms_locked.py`](code/envs/t1_sprint_arms_locked.py)
- [`code/utils/runner.py`](code/utils/runner.py) (só ramos DUP/LOSS)
- [`code/utils/symmetry.py`](code/utils/symmetry.py)

Não mexer: `envs/t1.py`, `train.py`, `play.py`, URDF/XML. O `tests/test_sameness.py` falha se um imutável divergir do root.

Depois de editar o YAML de treino, copia o snapshot para `iterations/<date>_vN/T1_sprint_arms_locked.yaml`.

## YAML (primeira escolha)

| Queres… | Campo | Valor desta passagem | Notas |
|---------|-------|----------------------|-------|
| Começar mais perto do T0 | `commands.lin_vel_start` | 0.9 | Teto inicial de `vx`. Não saltar para ≥2.0 no dia 1. |
| Piso do comando | `commands.min_lin_vel_x` | 0.5 | Overlap com o walk T0 (≤1 m/s). |
| Teto absoluto | `commands.lin_vel_x[1]` | 3.5 | `curr_v_max` nunca passa daqui. |
| Subir mais depressa | `commands.update_rate` | 0.1 | Fórmula T3: `v += rate * (v_hi - v)`. 0.15 é mais agressivo. |
| Facilitar o climb | `commands.lin_vel_x_toler` | 0.35 | 0.25 (v3) emperra. Não passar ~0.40. |
| Cadência | `commands.gait_frequency` | [1.5, 3.5] | Não começar já em [2.8, 3.5] a partir do T0. |
| Passada | `control.action_scale` | 1.2 | v8. 1.4 é o passo seguinte, não o primeiro. |
| Menos travão | `rewards.scales.action_rate` / `power` | -0.7 / -1e-3 | v6. Voltar a -1.0 / -2e-3 se ficar ruidoso. |
| Duração | `basic.max_iterations` | 15000 | Uma passagem 0.9→2.2 precisa disto. |
| Checkpoint | CLI `--checkpoint` | T0 `model_10000.pth` | Deixar `basic.checkpoint: null` no YAML. |
| Simetria | CLI `--symmetry` ou `algorithm.symmetry.method` | `none` | `none` / `dup` / `loss` / `phase`. **Um** método por run. |
| Peso L_sym | `algorithm.symmetry.loss_weight` | 4.0 | Só entra se `method=loss`. |

Chaves `lin_vel_levels`, `ang_vel_levels`, `lin_vel_x_resolution` existem só porque o `T1.__init__` as lê. **Esta task não usa o grid 2D.** Mudá-las não muda o `vx`.

## Código (só se o YAML não chegar)

Tudo em [`code/envs/t1_sprint_arms_locked.py`](code/envs/t1_sprint_arms_locked.py):

| Função | O que faz |
|--------|-----------|
| `next_v_max` | `curr + rate * (v_hi - curr)`, teto `v_hi`. Testada em `tests/test_curriculum.py`. |
| `clamp_vx` | Mantém o comando na janela `[v_min, v_max]`. |
| `_resample_curriculum_commands` | Amostra `vx ~ U(curr_v_min, curr_v_max)`. **Não** chama o pai. |
| `_update_curriculum` | Se o episódio sobrevive e o tracking acerta, sobe `curr_v_max`. |
| `_apply_sprint_command_constraints` | `vy = yaw = 0`; no play usa `--play_lin_vel_x`. |
| `_sync_curriculum_metrics` | `curriculum/max_lin_vel_level` no TensorBoard = `curr_v_max`. |
| `_phase_enabled` / `step` / `_compute_observations` | PHASE: espelha action e obs se `gait_process >= 0.5`. |

DUP e LOSS estão no runner, não no env. O runner **não** ramifica em `phase`.

O runner **não** guarda `curr_v_max` no `.pth`. Um resume recomeça o teto em `lin_vel_start`; a policy já mais rápida volta a subir.

## Não fazer

- Ligar `rewards.scales.feet_air` (v7 colapsou).
- Pôr `lin_vel_start` ou `min_lin_vel_x` ≥ 2.0 no instante 0 (salto T0 → 2 m/s).
- Editar `envs/t1.py` para “corrigir o grid” — o 1D está nesta classe de propósito.
- Treinar do zero (`--checkpoint` vazio): v1–v3 falharam.
- Usar o YAML da v9 do `T1_sprint` (`min` 2.2, start ~3.0, `action_scale` 1.4).
- Ligar dois métodos ao mesmo tempo (`dup`+`phase`, etc.).

## Nova reward

1. Em `envs/t1.py` **não**. Acrescenta `_reward_<nome>` na classe `T1_sprint_arms_locked` (ou reutiliza um `_reward_*` que já existe no `T1`).
2. Põe `rewards.scales.<nome>: <peso>` no YAML.
3. Scales `0` são ignorados (`T1._prepare_reward_function`).

## Outro checkpoint T0

```sh
python3 train.py --task=T1_sprint_arms_locked \
  --checkpoint=/app/logs/<date-time-T0>/nn/model_10000.pth \
  --headless True
```

Tem de ser 12 DoF / 47 obs / 12 actions (`T1_locomotion`). Um ckpt T6/T7 (23 DoF) não carrega nesta task.

## Conferir que não partiste a cópia

```sh
cd T1_sprint_arms_locked/code
python3 tests/test_sameness.py
python3 tests/test_curriculum.py
python3 tests/test_symmetry.py
python3 tests/test_method_isolation.py
```
