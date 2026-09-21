# MLU-DPGEN 工作流全细节文档

分支 `mlu`（基于 wccc-phys/dpgen master `0b9acec`）。本文档是磁性 Fe DP+U 势主动学习工作流的唯一权威参考，覆盖数据约定、各阶段逻辑、全部新增参数、逐代 U_init 演变、机器配置与迁移清单。

## 1. 工作流总览

```
┌─ 第 0 步：相区定义 ────────────────────────────────────────────┐
│ regions.json：相(bcc/fcc/hcp/liq…) × T/P 区间 × 磁性标注        │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ 初始数据 ─────────────────────────────────────────────────────┐
│ dpgen mlu uinit      pw.x+hp.x 在 (T,P) 点的 U 自洽循环 → u_tp   │
│ (提交集群, 跑完收集)  dpgen mlu uinit-collect → uinit/u_tp.json   │
│ dpgen mlu aimd       每相区×T×P 生成 pw.x MD 粗采样任务           │
│ dpgen mlu pick       pymatgen MSD 振幅加权选帧 → 未标定初始集     │
│ (初始集送入 dpgen run 的 02.fp 标定链，磁性帧逐帧 HP 重算 U)      │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌─ dpgen run 迭代循环（每代 9 任务）──────────────────────────────┐
│ 00.train   委员会 DP+U（异构模板可）+ MLU 双轨（有 uparam 起）   │
│ 01.model_devi  LAMMPS 全相区探索（npt-x/y/z、mlu_model 预测 U，  │
│                 偏差仅记录不筛选）                                │
│ [make_fp]  mlu_select 委员会描述符并集筛选 − 已标定 → 候选        │
│ 02.fp      pwscf MLU 链：磁性=scf→hp→scf_new，非磁性=scf          │
│ [post_fp]  直写 npy：fparam/aparam/uparam                        │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
   dpgen collect --mlu   汇总全部数据（f/a/uparam 完整保留+去重）
```

## 2. 数据约定（全链不变）

| 文件 | 形状 | 含义 | 写入处 | 消费处 |
|---|---|---|---|---|
| `fparam.npy` | (n,1) | 帧温度 K | post_fp（job.json ele_temp）/ `dpgen mlu pick` | DP+U 拟合输入、MLU 输入、QE degauss=T×6.3336e-6 Ry |
| `aparam.npy` | (n,natoms,1) | 原子温度 K（use_ele_temp=2） | post_fp | DP+U 拟合输入 |
| `uparam.npy` | (n,1) 帧模式 / (n,natoms) 原子模式 | Hubbard U eV | HP 链标定输出 | DP+U 拟合输入；**MLU 的标签** |
| `virial.npy` | (n,9) | eV（W=−σV） | 标定输出 | 训练 |

- **dpdata 全程不接触这三个文件**：mlu_select.load_system / post_fp_pwscf_mlu / collect --mlu 均为 npy 直读直写。
- 磁性标注：`magnetic_sys_idx`（sys_configs 索引列表，支持 `"0..3"` 区间）。fp 任务目录名 `task.SYS.NNNNNN`、探索任务 `task.SYS.NNN` 的 SYS 即来源相索引；初始数据系统按目录名推断（含 bcc/hcp/fcc/mag 关键词视为磁性，可用 `magnetic_init_sys` 显式覆盖）。
- 赝势/ecut/kspacing 全链一致（示例：`Fe.pbe-szn-kjpaw_psl.1.0.0.UPF`、ecutwfc=80/ecutrho=400、kspacing 由 `qe_hp.compute_kmesh`）。

## 3. 逐代 U_init 演变（磁性帧标定的种子）

| 代 | U_init 来源 | 说明 |
|---|---|---|
| 第 0 代（无 MLU） | `fp_hubbard.u_init_source="auto"`：u_tp 表（`dpgen mlu uinit` 产物，最近 T/P 匹配）> 常数 `u_init`（默认 1.0 eV） | AIMD 磁性相的 HUBBARD 卡同样用 u_tp 最近值 |
| ≥ 出现 MLU 模型起 | MLU 预测（`00.train/mlu_model*`，DeepEval 调用，帧模式给标量、原子模式给每原子） | `_resolve_u_init` 优先级：mlu > u_tp > const；`u_init_source` 可强制 `"mlu"/"tp"/"const"` |

HP 链随后重算精确 U——U_init 只影响收敛速度，不影响最终标签。

## 4. 各阶段细节

### 4.1 初始管线（`dpgen mlu` 子命令）

- **`uinit PARAM [MACHINE]`**：每 (T,P) 点一个任务 `uinit/T****K_P****bar/`，远程 `uinit_loop.py` 迭代 scf(U)→hp→U_new 直到 |ΔU|<tol（默认 0.02 eV，max_iter 5）。NPT（P>0）通过在 &CONTROL 注入 `cell_dynamics='pr'`/`press`/`cell_dofree='all'`。`run.sh` 由机器命令驱动（默认 pw.x/hp.x，可用 fp_pw_command/fp_hp_command 覆盖）。
- **`uinit-collect PARAM`**：汇总 `u_result.json` → `uinit/u_tp.json`（即 fp_hubbard.u_tp 的直接输入）。
- **`aimd PARAM [MACHINE]`**：`aimd/<region>/T****K_P****bar/` 生成 `calculation='md'` 输入：NVT（nose-hoover/langevin）或 NPT（PR cell dynamics）；磁性相 nspin=2+starting_magnetization+HUBBARD(u_tp 最近值)；dt 由 fs→a.u. 换算（1 fs ≈ 20.67 a.u.）。
- **`pick PARAM`**：ASE espresso-out 读取 pw.x MD 轨迹 → pymatgen Structure → 分数坐标最小镜像去卷绕 MSD；振幅 A=后 1/4 段 sqrt(MSD)；每轨迹选帧数 ∝ A（总预算 budget，保底 1），帧内均匀间隔（跳过 skip 步平衡段）；输出 `init_data/<region>.T.P/set.000/{box,coord,fparam}.npy` + `manifest.json`（含每轨迹振幅/磁性/T/P，供后续标定路由）。

### 4.2 训练（00.train）

- **委员会**：`default_training_param` 为 dict 时所有成员共享（各自动随机种子）；为 list 时每个成员独立模板（**描述子可异构**，长度必须 = numb_models）。
- **MLU 双轨**：设 `mlu_training_param`（fitting_net.type="mlu"，loss.type="mlu"，intensive:null 自动探测帧/原子模式）→ 每代在 `00.train/mlu/` 用携带 uparam.npy 的训练系统子集重训 MLU；post_train 把冻结模型暴露为 **`00.train/mlu_model*`**（绝不能叫 graph.* —— 探索阶段按 graph* 收集委员会能量模型）。空迭代/跳训时 copy_model 继承上一代 MLU。

### 4.3 探索（01.model_devi）

- 新系综 **npt-x / npt-y / npt-z**：`fix npt temp ... x|y|z ${PRES} ${PRES} ${TAU_P}` 单轴压缩。
- U 注入（互斥二选一）：
  - `mlu_model`（推荐，"auto"= 最新冻结 MLU）：`pair_style deepmd <graphs> ... mlu_model ../mlu_model.pth` 每步预测 U；模型文件由 make_model_devi 链接进 01.model_devi/。
  - `lmp_uparam`：常数帧 U（`uparam <U>` 关键字）。
  - 两者均可在 jdata 全局或 model_devi_jobs 逐代覆盖；fparam/aparam 温度沿用 use_ele_temp 1/2。
  - **需要 deepmd-kit fork 编译的 LAMMPS**（上游 lmp 无这些关键字）。
- 步数逐代调度：model_devi_jobs 每项的 `nsteps`（首代可短）。
- **模型偏差只记录不筛选**（你的第 4 点决策）：model_devi.out 照常生成。

### 4.4 筛选（make_fp → mlu_select）

`jdata["mlu_select"]` 存在时取代 trust-level：
1. **数据池** = init_data + 历代 02.fp/data.* + 本代探索帧（按磁性分两组独立筛选）。
2. 每个委员会模型对全池算描述符（DeepPot.eval_descriptor，f/aparam/uparam 按模型维度透传）。
3. 每系统两阶段收敛（粗扫 20…100% + 按 convergence_tolerance% 向下细化），指标：PC1-PC2 凸包覆盖/网格覆盖/KDE JS +（有标签系统）归一化 E/F/V Wasserstein；阈值全过即通过。
4. 逐模型最小子集 → **并集** → 指纹（box+coord, 1e-6）减去全部已标定与池内重复 → 候选。
5. 候选生成标准 fp 任务目录（conf.dump/job.json/POSCAR），任何 fp_style 可用；报告写 `01.model_devi/mlu_select_report.json`。

### 4.5 标定（02.fp，fp_style=pwscf + fp_hubbard）

- 非磁性：单步 scf（nspin_nm，degauss=帧温度）。
- 磁性：`run.sh` 链 pw.x scf(U_init)→`mlu_frame.py check-scf`→hp.x→`mlu_frame.py gen-scf-new`→pw.x scf_new→check；全部输出汇集到 `output`，成功标志 `CHAIN_DONE`（失败 `CHAIN_FAILED`；逐帧跳过用 if 块，无 mlu_model 的 continue bug）。
- 远程依赖：python3+numpy（`mlu_frame.py` 随迭代投递，自包含）。
- post_fp_pwscf_mlu：几何取自 pw.scf.in（含 Fe1..FeN 赝物种合并回元素），能量/力/应力取 scf_new.out（非磁性取 scf.out），U 取 pw.Hubbard_parameters.dat；CHAIN_DONE+最终收敛才收帧；直写 npy（fparam/aparam/uparam）。
- uparam 模式：`fp_hubbard.uparam_mode`="frame"（(n,1) 平均）或 "atomic"（(n,natoms) 每原子，QE 侧走赝物种布局）。

### 4.6 汇总（dpgen collect --mlu）

按系统索引（相分组不丢）合并 init + 全部迭代，f/a/uparam 逐帧保留，跨贡献者指纹去重（--no-dedup 关闭），set_size 分片，输出 `collect_report.json`。贡献者参数不一致（一个有一个没有）或 natoms 不同 → 明确报错拒绝合并。

## 5. 参数速查（新增部分）

### param.json（run）
| 键 | 位置 | 默认 | 说明 |
|---|---|---|---|
| `magnetic_sys_idx` | 顶层 | [] | 磁性相的 sys_configs 索引（支持 "0..3"） |
| `mlu_training_param` | 顶层 | 无 | MLU 模板（fitting type=mlu） |
| `lmp_uparam` / `mlu_model` | model_devi 或逐 job | 无 | 探索 U 源（互斥；mlu_model 支持 "auto"） |
| `mlu_select` | 顶层 | 无 | 并集筛选配置（method/tolerance/seed/阈值平铺键） |
| `fp_hubbard` | fp 段 | 无 | element/manifold/u_init/u_init_source/u_tp/uparam_mode/nspin*/starting_mag/nq*/aparam_uniform_fallback |
| `magnetic_init_sys` | 顶层 | [] | 初始数据系统名列表（磁性显式标注） |

### machine.json（fp 段新增）
| 键 | 默认 | 说明 |
|---|---|---|
| `fp_command` | — | MLU 模式下设为 `bash run.sh` |
| `fp_pw_command` | "pw.x" | run.sh 内 pw.x 命令（含 mpirun/-nk 等） |
| `fp_hp_command` | "hp.x" | run.sh 内 hp.x 命令 |

## 6. 迁移到计算平台的待办清单

容器内已完成代码+单元/集成测试（无重计算）。迁移后需按序验证：

1. **环境**：clone 本 fork（mlu 分支）+ `/personal/deepmd-kit` fork；`pip install -e` 到同一 venv + scikit-learn/scipy/matplotlib/pymatgen/ase；改 deepmd-kit 源后必须重装（venv 里是快照安装）。
2. **LAMMPS 编译**：用 deepmd-kit fork 的 `.devcontainer/build_cxx_mlu.sh`（含 uparam/mlu_model 关键字与 compute deepmd/u），spike 验证多模型 model_devi + mlu_model 同用。
3. **uinit 冒烟**：选 1 个 (T,P) 点跑通自洽循环，核对 u_tp.json 数值量级（Fe ~3–4 eV）。
4. **AIMD 冒烟**：1 相区短 MD（~100 步）验证 `dpgen mlu pick` 全链（espresso-out 解析→MSD→选帧）。
5. **标定链冒烟**：2 帧（磁性+非磁性）过 run.sh 链，核对 uparam 与 mlu_model 参考值（mlu_model datasets 有历史数据可回归）。
6. **端到端小 run**：2 相区（bcc 磁性+liq 非磁性）× 2 迭代 × fp_task_max≤10，全 9 任务+collect --mlu。
7. **回归基线**：test0/select_configs.py 在同一数据上对比 mlu_select 的指标口径（文档化差异：hull_coverage 此处定义为选中帧落入全池凸包的比例）。

## 7. 与上游/参考实现的差异备忘

- `npt-x/y/z` 只加了经典 MD 分支；PIMD 分支沿用上游白名单（会拒绝）。
- 凸包覆盖率语义与 test0 的 13 指标口径略有差异（见 §6.7）；JS 散度用直方图 KDE 近似（同 test0 实现）。
- `use_ele_temp` 文档已放开 vasp-only 限制（pwscf MLU 链同样消费）。
- mlu_select 的 `_infer_magnetic_from_name` 是初始系统名的兜底推断，正式相区请用 `magnetic_init_sys` 显式标注。
