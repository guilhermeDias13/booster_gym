# Como modificar T1_sprint_arms_not_locked

Primeiro o YAML. Só mexer no `.py` se o knob não existir.

Ficheiros que podes editar:

- [`code/envs/T1_sprint_arms_not_locked.yaml`](code/envs/T1_sprint_arms_not_locked.yaml)
- [`code/envs/t1_sprint_arms_not_locked.py`](code/envs/t1_sprint_arms_not_locked.py)
- [`code/utils/runner.py`](code/utils/runner.py) (DUP/LOSS e a mensagem do expand)
- [`code/utils/model.py`](code/utils/model.py) (expand 12 → 21)
- [`code/utils/symmetry.py`](code/utils/symmetry.py)

Não mexer: `envs/t1.py`, `train.py`, `play.py`, URDF/XML. O `tests/test_sameness.py` falha se um ficheiro partilhado divergir do locked ou se o URDF divergir do T7 head-fixed.

## YAML (primeira escolha)

Os knobs de velocidade são os do locked. Ver a tabela em [`T1_sprint_arms_locked/MODIFY.md`](../T1_sprint_arms_locked/MODIFY.md). Resumo desta passagem:

| Queres… | Campo | Valor |
|---------|-------|-------|
| Teto inicial de `vx` | `commands.lin_vel_start` | 0.9 |
| Piso | `commands.min_lin_vel_x` | 0.5 |
| Teto absoluto | `commands.lin_vel_x[1]` | 3.5 |
| Passada | `control.action_scale` | 1.2 |
| Simetria | `--symmetry` ou `algorithm.symmetry.method` | `none` / `dup` / `loss` / `phase` |
| Joint deviation | `--joint_deviation` ou `rewards.scales.joint_deviation` | `-0.` (desligado); `-0.5` para ligar |
| Duração | `basic.max_iterations` | 15000 |

Braços e cintura:

| Queres… | Campo | Valor desta passagem |
|---------|-------|----------------------|
| Pose com ação 0 | `init_state.default_joint_angles` | ombros 0.2 / ∓1.35, `Elbow_Yaw` ∓0.5 |
| Cintura mais rígida | `control.stiffness.Waist` / `damping.Waist` | 200 / 5 |
| Braços mais moles | `Shoulder` / `Elbow` | 20 / 0.5 |

`num_actions` tem de ficar 21 e `num_observations` 74. Mudar só um dos dois parte o espelho e o expand.

Joint deviation (arXiv 2502.16230, erro quadrático em vez do valor absoluto). Os pesos por grupo ficam no código (`joint_deviation_weights`): hip/joelho/tornozelo pitch 0.5, hip yaw/roll 0.1, o resto (braços, cintura, ankle roll) 0.2. O YAML só tem a escala global. Com `-0.` o termo é descartado em `_prepare_reward_function` e o treino fica igual ao de antes. `--joint_deviation -0.5` liga o termo.

## Checkpoint

```sh
python3 train.py --task=T1_sprint_arms_not_locked \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True
```

Tem de ser 12 DoF / 47 obs / 12 actions. O runner expande. Um ckpt já 21 ou 23 DoF não entra por este caminho (o `load_state_dict` estrito do expand não corre; o fallback `strict=False` deixa camadas de tamanho diferente de fora).

## Não fazer

- Soltar a cabeça neste YAML: isso é outro URDF (`T1_serial`, 23 DoF) e outro `_N_UPPER`.
- Ligar `feet_air`.
- Começar com `lin_vel_start` ≥ 2.0.
- Treinar sem o checkpoint T0.
- Ligar dois métodos de simetria ao mesmo tempo.

## Conferir

```sh
cd T1_sprint_arms_not_locked/code
python3 tests/test_sameness.py
python3 tests/test_curriculum.py
python3 tests/test_symmetry.py
python3 tests/test_method_isolation.py
python3 tests/test_expand.py
python3 tests/test_joint_deviation.py
```
