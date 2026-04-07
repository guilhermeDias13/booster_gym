# Spawn, determinismo e randomização (Booster Gym / T1)

Este documento descreve **onde** a pose e a posição inicial são definidas, **o que ainda variava** no código original além do YAML, e **como** obter reset repetível com mudanças **mínimas** no código (comportamento padrão igual ao upstream quando as chaves novas não existem).

**Implementação:** a lógica descrita nas secções 4–6 está aplicada em [`envs/t1.py`](../envs/t1.py) e espelhada em [`envs/t2.py`](../envs/t2.py). Os ficheiros [`envs/T1.yaml`](../envs/T1.yaml) e [`envs/T2.yaml`](../envs/T2.yaml) usam neste repositório `randomize_yaw: false`, `actuation_delay_steps: 0`, `noise: {}`, `randomization: {}` e `terrain.type: plane` para treino determinístico e plano; para voltar ao comportamento próximo do [upstream](https://github.com/BoosterRobotics/booster_gym), repor `randomize_yaw: true` (ou remover a chave), remover `actuation_delay_steps`, `terrain.type: trimesh`, e copiar os blocos `noise` e `randomization` completos do repositório oficial.

---

## 1. Postura e posição no YAML

Tudo começa em `init_state` em [`envs/T1.yaml`](../envs/T1.yaml):

| Campo | Papel |
|--------|--------|
| `pos` | Posição inicial do tronco (x, y, z) antes de somar origem do ambiente e correção de terreno. |
| `rot` | Quaternion (x, y, z, w) da base — usado quando o yaw **não** é randomizado (ver secção 5). |
| `randomize_yaw` | Se `true` (omissão = `true`): yaw uniforme a cada reset (original). Se `false`: usa `rot` em cada reset. |
| `lin_vel`, `ang_vel` | Velocidades iniciais da base. |
| `default_joint_angles` | Ângulos alvo das juntas (nome do DOF contém a chave, ex. `Knee_Pitch`); também servem de referência para o PD quando a ação é zero. |

**Postura em pé / joelhos:** ajuste apenas `default_joint_angles` (e, se necessário, `pos` em z) — não exige mudança de código.

---

## 2. O que o código faz no reset (fluxo)

No reset, [`envs/t1.py`](../envs/t1.py):

1. Copia `base_init_state` (montado a partir de `pos`, `rot`, `lin_vel`, `ang_vel` do YAML).
2. Soma `env_origins` em XY (cada ambiente paralelo tem a sua origem — ver secção 6).
3. Opcionalmente aplica `randomization.init_base_pos_xy` (se existir no YAML).
4. Ajusta z com a altura do terreno em `terrain_heights(xy)`.
5. Define a orientação da base: com `randomize_yaw: true` (padrão), **yaw aleatório**; com `false`, usa `rot` do YAML.
6. Opcionalmente aplica `randomization.init_base_lin_vel_xy`.

Ou seja: mesmo com YAML fixo, **o yaw aleatório** e **o atraso aleatório de atuação** faziam cada episódio diferente em orientação e fase do comando.

---

## 3. Ruído nas observações (`noise`)

Em `_compute_observations`, cada canal usa `apply_randomization(..., self.cfg["noise"].get(...))`. Se a chave não existe ou `noise: {}`, `.get` devolve `None` e **não há ruído** (comportamento já suportado por [`utils/utils.py`](../utils/utils.py)).

**Sim-to-real:** desligar ruído acelera debug mas reduz robustez; para deploy, o upstream usa ruído nas obs.

---

## 4. Randomização de domínio (`randomization`)

Parâmetros como `init_dof_pos`, `dof_stiffness`, `kick_lin_vel`, etc. são lidos com `.get(...)` — se ausentes, **não alteram** o tensor.

**Problema no original:** `_kick_robots` e `_push_robots` usavam `self.cfg["randomization"]["kick_interval_s"]` (e similares) com `[]`. Com `randomization: {}` vazio, isso gerava `KeyError`.

**Correção mínima (comportamento igual ao upstream se o YAML tiver intervalos):** no início de `_kick_robots`, se `randomization.get("kick_interval_s")` for `None`, `return`. Em `_push_robots`, se `push_interval_s` ou `push_duration_s` forem `None`, `return`. Assim `randomization: {}` desliga kicks/pushes sem crash.

---

## 5. Yaw fixo = usar `rot` do YAML

No original, após posicionar o corpo, o código **substituía** o quaternion por um yaw uniforme em `[0, 2π)`.

**Correção mínima:** só envolver esse bloco:

- Se `init_state.randomize_yaw` for **omitido** ou `true` → manter o código original (yaw aleatório).
- Se `init_state.randomize_yaw: false` → definir `root_states[env_ids, 3:7]` a partir de `init_state.rot`, replicado para todos os `env_ids`. O quaternion intermédio deve usar **o mesmo `dtype` e `device` que `root_states`** (por exemplo `to_torch(..., device=self.root_states.device).to(dtype=self.root_states.dtype)`) para evitar promover para double, CPU/GPU mismatch ou avisos em atribuições.

Default `True` preserva o comportamento do repositório oficial sem alterar YAML antigo.

---

## 6. Atraso de atuação (`delay_steps`)

Em `_reset_idx`, o original faz:

```python
self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], ...)
```

**Correção mínima:** se existir `control.actuation_delay_steps` (inteiro), usar esse valor para todos os ambientes resetados, com clamp em `[0, decimation - 1]`. Se a chave **não** existir, manter o `torch.randint` original.

---

## 7. Vários ambientes (`num_envs > 1`)

Cada ambiente tem `env_origins` diferente. “Sempre no mesmo sítio” é **por ambiente** (mesmo offset relativo ao tile), não o mesmo ponto global para todos os 2048/4096 robôs.

---

## 8. Terreno plano

Com `terrain.type: plane`, a altura de solo é consistente no tile; com trimesh, `terrain_heights(xy)` pode variar. Para treino só plano, defina `terrain.type: "plane"` no YAML (resto da secção `terrain` pode ficar; campos específicos de trimesh são ignorados em [`base_task`](../envs/base_task.py) conforme o tipo).

---

## Referência de código (já integrado em `t1.py` / `t2.py`)

Os trechos abaixo correspondem ao que está no repositório; servem de referência se precisar de reverter ou comparar com o upstream.

### `envs/t1.py` — `_reset_idx` (substituir só a linha do `delay_steps`)

**Antes:**

```python
        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
```

**Depois:**

```python
        if "actuation_delay_steps" in self.cfg["control"]:
            d = int(self.cfg["control"]["actuation_delay_steps"])
            d = max(0, min(d, self.cfg["control"]["decimation"] - 1))
            self.delay_steps[env_ids] = d
        else:
            self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
```

### `envs/t1.py` — `_reset_root_states` (substituir só o bloco do quaternion)

**Antes:**

```python
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
```

**Depois:**

```python
        if self.cfg["init_state"].get("randomize_yaw", True):
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
                torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
            )
        else:
            quat = to_torch(self.cfg["init_state"]["rot"], device=self.root_states.device).to(
                dtype=self.root_states.dtype
            )
            self.root_states[env_ids, 3:7] = quat.unsqueeze(0).expand(len(env_ids), -1)
```

(`to_torch` já está importado no ficheiro. Alinhar a `root_states` evita bugs sutis de tipo/dispositivo na atribuição.)

### `envs/t1.py` — `_kick_robots`

**Antes** (primeiras linhas do método):

```python
    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
```

**Depois:**

```python
    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        kick_interval_s = self.cfg["randomization"].get("kick_interval_s")
        if kick_interval_s is None:
            return
        if self.common_step_counter % np.ceil(kick_interval_s / self.dt) == 0:
```

### `envs/t1.py` — `_push_robots`

**Antes** (início do método até ao `elif`):

```python
    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
```

**Depois:**

```python
    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        push_interval_s = self.cfg["randomization"].get("push_interval_s")
        push_duration_s = self.cfg["randomization"].get("push_duration_s")
        if push_interval_s is None or push_duration_s is None:
            return
        if self.common_step_counter % np.ceil(push_interval_s / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(push_interval_s / self.dt) == np.ceil(push_duration_s / self.dt):
```

### `envs/t2.py`

Mesma lógica que `t1.py` (ficheiro espelhado).

---

## YAML neste repositório (`T1.yaml` / `T2.yaml`)

`T1.yaml` e `T2.yaml` incluem, para reset determinístico e terreno plano:

- `init_state.randomize_yaw: false`
- `control.actuation_delay_steps: 0`
- `noise: {}` e `randomization: {}`
- `terrain.type: plane`

Para comportamento **próximo do upstream oficial**, repor `randomize_yaw: true` (ou remover), remover `actuation_delay_steps`, `terrain.type: trimesh`, e os blocos `noise` + `randomization` completos do [BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym).

---

## 7. Reset vs time-out e logging no TensorBoard

Em `_check_termination` (`t1.py` / `t2.py`):

- **`reset_buf`** (fim de episódio com reset físico): contato em `terminate_contacts_on`, velocidade da raiz acima de `terminate_vel`, altura da base abaixo de `terminate_height`, ou timeout longo (`episode_length_s`).
- **`time_out_buf`** (truncagem para o PPO / bootstrap): inclui o timeout longo **e** o marco `episode_length_buf == cmd_resample_time` (reamostragem de comando). Este último **não** entra em `reset_buf`: o episódio continua, só se marcam truncações para o algoritmo.

Em cada passo, `extras["time_outs"]` reflete o `time_out_buf` atual (para o [`Runner`](../utils/runner.py) gravar no buffer). Os dicionários `extras["term_causes"]`, `extras["term_primary"]` e `extras["trunc_cmd_resample"]` descrevem causas de **reset** e a fração de envs em reamostragem de comando.

Com `runner.log_termination_reasons: true` (padrão no código se a chave faltar), o [`Recorder`](../utils/recorder.py) regista por iteração de treino:

| Escalar | Significado |
|---------|-------------|
| `termination/frac_contact`, `frac_vel`, `frac_height`, `frac_episode_timeout` | Média das flags (0–1) nos ambientes que fizeram **reset** na janela do rollout; podem sobrepor-se se várias condições forem verdadeiras no mesmo passo. |
| `termination/primary/contact`, `…/vel`, `…/height`, `…/episode_timeout` | Fração dos resets onde a causa “primária” foi essa, com prioridade: contacto > velocidade > altura > timeout de episódio (soma ~1 sobre os resets contabilizados). |
| `truncation/frac_cmd_resample` | Média ao longo do rollout da fração de envs com `episode_length_buf == cmd_resample_time` (truncação por comando, **não** reset). |

Defina `runner.log_termination_reasons: false` no YAML para desligar estes escalares.

---

## Resumo

| Objetivo | Onde |
|----------|------|
| Mesma postura | `init_state.default_joint_angles` |
| Mesma orientação no reset | `randomize_yaw: false` + `rot` |
| Sem ruído nas obs | `noise: {}` ou chaves ausentes |
| Sem domain randomization / kicks / pushes | `randomization: {}` (o código trata `kick_interval_s` / `push_*` ausentes) |
| Sem jitter de atraso no comando | `control.actuation_delay_steps` definido |
| Plano horizontal | `terrain.type: plane` |
| Motivos de reset no TensorBoard | `runner.log_termination_reasons: true` (ver secção 7) |

**Aviso:** treino com spawn e domínio muito fixos reduz diversidade; é útil para hipóteses e debug, mas tende a piorar generalização e sim-to-real em relação ao paper.
