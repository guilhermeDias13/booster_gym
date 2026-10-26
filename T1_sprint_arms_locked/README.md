# T1_sprint_arms_locked

Uma passagem: caminhada T0 → corrida reta **> 2 m/s** no `T1_locomotion` (12 DoF, braços fixos).

Stack isolada em [`code/`](code/). Não altera `envs/t1.py` nem `envs/T1_sprint.yaml` do repo.

- O que mudou vs o root: [`CHANGES.md`](CHANGES.md)
- Como alterar knobs: [`MODIFY.md`](MODIFY.md)
- Iteração atual: [`iterations/2026-09-20_v1/`](iterations/2026-09-20_v1/)

## Porquê esta task

O `T1_sprint` histórico precisou da escada v4→v5→v6→v8 porque o curriculum 2D **congela `vx` em `min_lin_vel_x`**. Aqui o comando é 1D: `vx ~ U(0.5, curr_v_max)` e `curr_v_max` sobe de 0.9 para 3.5 quando o tracking acerta.

## Treino

Correr **a partir de** `T1_sprint_arms_locked/code/`. Default `--symmetry none` (mesma passagem T0 → >2 m/s). Repetir com `dup`, `loss` ou `phase`:

```sh
cd T1_sprint_arms_locked/code
python3 train.py --task=T1_sprint_arms_locked --symmetry none \
  --checkpoint=/app/logs/2026-08-21-14-44-42/nn/model_10000.pth \
  --headless True
# repetir com --symmetry dup | loss | phase
```

15k iterações. Checkpoint T0: run `2026-08-21-14-44-42`, `model_10000.pth`.

| Método | Onde | Efeito |
|--------|------|--------|
| `none` | — | baseline |
| `dup` | runner | surrogate também em `M(s), M(a)` |
| `loss` | runner | `w * L_sym`, `w=4.0` |
| `phase` | env | espelha obs/act se `gait >= 0.5` |

## Eval

```sh
cd T1_sprint_arms_locked/code
python3 play.py --task=T1_sprint_arms_locked \
  --checkpoint=logs/<date-time>/nn/model_15000.pth \
  --play_lin_vel_x 2.0 \
  --num_envs=1
```

Headless + vídeo no Vast:

```sh
xvfb-run -a python3 play.py --task=T1_sprint_arms_locked \
  --checkpoint=logs/<date-time>/nn/model_15000.pth \
  --play_lin_vel_x 2.0 --num_envs=1 --headless True
```

## TensorBoard

```sh
tensorboard --logdir T1_sprint_arms_locked/code/logs
```

| Tag | Bom sinal |
|-----|-----------|
| `curriculum/max_lin_vel_level` | sobe e **passa 2.0** (`curr_v_max`) |
| `tracking/cmd_vx` vs `tracking/vel_vx` | curvas próximas |
| `tracking/abs_err_vx` | desce (alvo abaixo de ~0.35) |
| `episode/tracking_lin_vel_x` | alto / estável |
| `episode/stand_still` | baixo |

## Go / no-go

- **Go:** linha reta, `vel_vx > 2.0` estável, sem cair. Flight phase **não** é critério.
- **Cai cedo:** baixar `action_scale` para 1.1 ou `lin_vel_start` para 0.7 — ver [`MODIFY.md`](MODIFY.md).
- **`curr_v_max` parado abaixo de 2:** afrouxar `lin_vel_x_toler` (máx. 0.40) ou confirmar que o checkpoint T0 carregou.

## Archive (vídeo + logs)

Uma pasta por método: `BASE` (`--symmetry none`), `dup`, `loss`, `phase`.

```powershell
scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new `
  root@<HOST>:/app/videos/<nome>.mp4 `
  .\T1_sprint_arms_locked\iterations\<iter>\<BASE|dup|loss|phase>\videos\

scp -P <PORT> -i .\key -o StrictHostKeyChecking=accept-new -r `
  root@<HOST>:/app/logs/<date-time>/ `
  .\T1_sprint_arms_locked\iterations\<iter>\<BASE|dup|loss|phase>\logs\
```

(Se o cwd no Vast for `T1_sprint_arms_locked/code`, os logs ficam em `logs/<date-time>/` relativamente a essa pasta.)

## Testes de sameness

```sh
cd T1_sprint_arms_locked/code
python3 tests/test_sameness.py
python3 tests/test_curriculum.py
python3 tests/test_symmetry.py
python3 tests/test_method_isolation.py
```
