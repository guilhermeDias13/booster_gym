# CHANGES — T1_sprint_arms_locked vs repo root

Cópia isolada do root no commit em [`code/SOURCE_COMMIT`](code/SOURCE_COMMIT). Hashes: [`_precopy_hashes.txt`](_precopy_hashes.txt).

## Ficheiros novos (o que é diferente)

| Ficheiro | vs root | Porquê |
|----------|---------|--------|
| [`code/envs/t1_sprint_arms_locked.py`](code/envs/t1_sprint_arms_locked.py) | novo | Curriculum 1D: `vx ~ U(curr_v_min, curr_v_max)`, sobe o teto no sucesso. O `T1_sprint` do root congela `vx` em `min_lin_vel_x`. |
| [`code/envs/T1_sprint_arms_locked.yaml`](code/envs/T1_sprint_arms_locked.yaml) | novo | 15k iters; partida T0 (`min` 0.5, `lin_vel_start` 0.9, faixa [0.5, 3.5]); `action_scale` 1.2; sem `feet_air`. |
| [`code/envs/__init__.py`](code/envs/__init__.py) | diferente | Só regista `T1_sprint_arms_locked` (o root lista T0–T6). |
| [`code/tests/test_sameness.py`](code/tests/test_sameness.py) | novo | Imutáveis = root; meshes = T7. |
| [`code/tests/test_curriculum.py`](code/tests/test_curriculum.py) | novo | Prova que o teto sobe e que `vx` não fica no mínimo. |
| [`code/utils/symmetry.py`](code/utils/symmetry.py) | novo (cópia T8) | Mapas 12-DoF + `L_sym` / DUP helpers. |
| [`code/utils/runner.py`](code/utils/runner.py) | **editável** | `--symmetry`, ramo DUP (surrogate + guardas NaN) e LOSS. Partiu da cópia root. |
| [`code/tests/test_symmetry.py`](code/tests/test_symmetry.py) | novo | Involução e `L_sym`. |
| [`code/tests/test_method_isolation.py`](code/tests/test_method_isolation.py) | novo | none/loss/phase não dobram o batch; só dup o faz. |

## Cópia byte-igual ao root (não mudar)

`train.py`, `play.py`, `envs/t1.py`, `envs/base_task.py`, `utils/model.py`, `utils/buffer.py`, `utils/recorder.py`, `utils/terrain.py`, `utils/utils.py`, `utils/__init__.py`, `resources/T1/T1_locomotion.urdf`, `resources/T1/T1_locomotion.xml`.

`utils/runner.py` **já não** é imutável: precisa dos ramos DUP/LOSS. O hash original continua em `_precopy_hashes.txt`.

## Meshes

`code/resources/T1/meshes/*.STL` copiados de [`T7/code/resources/T1/meshes`](../T7/code/resources/T1/meshes) (o root não tem essa pasta). Hash conferido em `tests/test_sameness.py`.

## O que de propósito **não** se copiou

- evaluate / `run_metadata` / `--cell` do T8 — não fazem falta nesta passagem.
- `envs/T1_sprint.yaml` v9 — start ~3 m/s + `action_scale` 1.4 é push a partir da v8, não do T0.

## Motivo do curriculum novo

No `T1_sprint` do root, o resample trata a **coluna** do grid como `lin_vel` e o update expande a **linha** (yaw), que depois é forçado a 0. O `start_level` não reescreve `commands[:, 0]`. Resultado: `vx = min_lin_vel_x` durante a run inteira — daí a escada manual v4→v8.

Esta classe **não chama** `T1._resample_curriculum_commands`. Amostra e sobe `curr_v_max` como o T3.
