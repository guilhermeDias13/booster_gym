# T1_sprint_arms_not_locked

Uma passagem: caminhada T0 → corrida reta **> 2 m/s** com braços e cintura livres. Cabeça fixa (`T1_serial_head_fixed`, 21 DoF).

Stack isolada em [`code/`](code/). Não altera `T1_sprint_arms_locked` nem o root.

- O que mudou vs o locked: [`CHANGES.md`](CHANGES.md)
- Como alterar knobs: [`MODIFY.md`](MODIFY.md)
- Treinos: [`iterations/sem_joint_dev/`](iterations/sem_joint_dev/notes.md) (sem o termo), [`iterations/joint_dev/`](iterations/joint_dev/notes.md) (`none` e `dup` a `-0.5`) e [`iterations/joint_dev_minus_1/`](iterations/joint_dev_minus_1/notes.md) (os quatro métodos a `-1`)

## Porquê esta task

O curriculum 1D é o mesmo do [`T1_sprint_arms_locked`](../T1_sprint_arms_locked/README.md): `vx ~ U(0.5, curr_v_max)`, teto a subir de 0.9 para 3.5. A diferença é o corpo: 8 DoF de braço + `Waist` + 12 de perna. A cabeça (`AAHead_yaw`, `Head_pitch`) continua `fixed`.

O checkpoint T0 é 12 DoF. O runner expande os pesos: pernas copiadas, braços e cintura a zero (ação 0 = pose default).

## Treino

Correr **a partir de** `T1_sprint_arms_not_locked/code/`. Default `--symmetry none`. Repetir com `dup`, `loss` ou `phase`:

```sh
cd T1_sprint_arms_not_locked/code
python3 train.py --task=T1_sprint_arms_not_locked --symmetry none \
  --checkpoint=/app/checkpoints/T0_model_10000.pth \
  --headless True
# repetir com --symmetry dup | loss | phase
```

15k iterações. O log deve dizer que o checkpoint 12 DoF foi expandido para a policy 21 DoF.

| Método | Onde | Efeito |
|--------|------|--------|
| `none` | — | baseline |
| `dup` | runner | surrogate também em `M(s), M(a)` |
| `loss` | runner | `w * L_sym`, `w=4.0` |
| `phase` | env | espelha obs/act se `gait >= 0.5` |

## Eval

```sh
cd T1_sprint_arms_not_locked/code
python3 play.py --task=T1_sprint_arms_not_locked \
  --checkpoint=logs/<date-time>/nn/model_15000.pth \
  --play_lin_vel_x 2.0 \
  --num_envs=1
```

Headless + vídeo no Vast:

```sh
xvfb-run -a python3 play.py --task=T1_sprint_arms_not_locked \
  --checkpoint=logs/<date-time>/nn/model_15000.pth \
  --play_lin_vel_x 2.0 --num_envs=1 --headless True
```

## TensorBoard

| Tag | Bom sinal |
|-----|-----------|
| `curriculum/max_lin_vel_level` | sobe e **passa 2.0** (`curr_v_max`) |
| `tracking/cmd_vx` vs `tracking/vel_vx` | curvas próximas |
| `tracking/abs_err_vx` | desce (alvo abaixo de ~0.35) |

## Go / no-go

- **Go:** linha reta, `vel_vx > 2.0` estável, sem cair. Braços podem oscilar; não é critério de falha se o tronco não cai.
- **Cai cedo:** baixar `action_scale` para 1.1 ou `lin_vel_start` para 0.7 — ver [`MODIFY.md`](MODIFY.md).
- **`curr_v_max` parado abaixo de 2:** afrouxar `lin_vel_x_toler` (máx. 0.40) ou confirmar a linha `Expanded T1 12-DoF...`.

## Archive (vídeo + logs)

Uma pasta por variante (`sem_joint_dev`, `joint_dev` ou `joint_dev_minus_1`) e, dentro, uma por método: `BASE` (`--symmetry none`), `dup`, `loss`, `phase`. O mp4 e o conteúdo da run ficam direto em `videos/` e `logs/`, sem a pasta `logs/<date-time>/` no meio. Qualquer treino com `--joint_deviation -1` vai para `joint_dev_minus_1`; as instruções estão em [`iterations/joint_dev_minus_1/notes.md`](iterations/joint_dev_minus_1/notes.md).

```powershell
scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new `
  root@<HOST>:/app/videos/<nome>.mp4 `
  .\T1_sprint_arms_not_locked\iterations\<sem_joint_dev|joint_dev|joint_dev_minus_1>\<BASE|dup|loss|phase>\videos\

scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new -r `
  root@<HOST>:/app/logs/<date-time>/. `
  .\T1_sprint_arms_not_locked\iterations\<sem_joint_dev|joint_dev|joint_dev_minus_1>\<BASE|dup|loss|phase>\logs\
```

## Testes

```sh
cd T1_sprint_arms_not_locked/code
python3 tests/test_sameness.py
python3 tests/test_curriculum.py
python3 tests/test_symmetry.py
python3 tests/test_method_isolation.py
python3 tests/test_expand.py
```
