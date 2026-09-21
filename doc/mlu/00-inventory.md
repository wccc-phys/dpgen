# MLU-DPGEN 现状盘点（M0）

日期：2026-09-22　分支：`mlu`（基于 upstream master `0b9acec`，即 fork wccc-phys/dpgen 的干净 master，尚无自定义提交）

本文件盘点三个代码库在 MLU（磁性 DP+U 势主动学习）工作流中的现状与职责，作为后续所有改动的基线。

## 1. 代码库职责

| 代码库 | 位置 | 职责 | 状态 |
|---|---|---|---|
| dpgen（本仓库） | `/personal/dpgen`，分支 `mlu` | 主动学习编排：迭代循环、任务生成、机器调度、数据收集 | upstream master 原版，所有 MLU 改动在本分支进行 |
| deepmd-kit fork | `/personal/deepmd-kit`（origin wccc-phys/deepmd-kit） | DP+U 模型（uparam 输入）与 MLU 模型（U 预测） | 已提交 uparam 层（`72484546`）+ 未提交 MLU 层；本工作流不改此库 |
| mlu_model | `/personal/wangc/fe/mlu/mlu_model/qe_dfpt.py` | QE HP 标定链参考实现（QE 7.4.1 集群验证过） | 逻辑移植进本仓库 `dpgen/generator/lib/qe_hp.py`，原库不动 |

## 2. deepmd-kit fork 提供的能力（不改，仅调用）

### DP+U 模型（能量模型）
- `fitting_net` 新键：`default_uparam`（float；设置后 uparam 维度隐式为 1）+ `uparam_mode`（`"frame"` / `"atomic"`）
- U 拼接进拟合网络输入：`concat([descriptor, fparam?, uparam?, aparam?])`（`general_fitting.py`）
- 数据文件：`set.*/uparam.npy`，frame 模式 `(n,1)`、atomic 模式 `(n,natoms)`
- 推断：`DeepPot.eval(..., uparam=...)`、`eval_descriptor(..., uparam=...)`

### MLU 模型（U 预测器）
- `fitting_net.type = "mlu"`（var_name=uparam，从结构+温度预测 U）、`loss.type = "mlu"`、`intensive: null` 自动探测 frame/atomic
- 组合推理 `DeepDpU(dp_model, mlu_model).eval(...)`：MLU 预测 U → 作为 uparam 喂给 DP+U
- C++ `DeepMLU` + LAMMPS：`pair_style deepmd ... mlu_model <path>`（每步预测 U）、`uparam U1 [U2...]`、`uparam_from_compute/fix`；`compute deepmd/u`、`compute deepmd/uparam/dedn`

## 3. dpgen 基线现状（upstream master `0b9acec`）

### 已有、直接复用
- 迭代循环 `run_iter`：每代 9 个任务（make/run/post × train/model_devi/fp），record.dpgen 断点续跑
- 委员会训练：`numb_models` + 新增跨架构委员会（`_get_committee_model_sections`，PR #1971）；多模型 → model_devi
- `use_ele_temp` 0/1/2：LAMMPS 注入 `fparam/aparam ${ELE_TEMP}`（帧温度/原子温度），post_fp_vasp 把 `job.json:ele_temp` 写成 `fparam.npy`/`aparam.npy`
- `sys_configs` + `sys_idx`：多相区探索任务划分
- 逐代可变参数：`model_devi_jobs` 每代独立的 trust level / ensemble / T / P（`cur_job` 覆盖 jdata）
- fp_style `pwscf`：`make_fp_pwscf` 生成 QE scf 输入（dpdata 读 POSCAR），`post_fp_dpdata` 用 dpdata qe/pw/scf 收数据
- `model_devi_jobs.rev_mat`、`steps`（每 job 的 LAMMPS 步数已经支持逐 job 设置）

### 缺失、需要实现（对应里程碑）
| 功能 | 现状 | 里程碑 |
|---|---|---|
| 系综 `npt-x/y/z`（单轴压缩） | `make_lammps_input` 仅 iso/aniso/tri，其余 raise | M1 |
| LAMMPS uparam / mlu_model 注入 | pair_style 行无相关关键字 | M1 |
| QE fparam（温度→degauss） | pwscf.py 无处理 | M2 |
| QE uparam（HUBBARD 卡） | 无 | M2 |
| 磁性 HP 三步标定链 | 无 | M2 |
| 委员会异构描述子模板 | 委员会共享同一 input 模板（仅种子不同） | M4 |
| MLU 双轨训练 | 无 | M4 |
| 描述符多样性筛选（替代 trust-level） | 仅 model deviation trust level 筛选 | M5 |
| collect 保留 fparam/aparam/uparam | `collect_data` 不带参数 | M6 |
| init 阶段 AIMD/MSD/相区定义 | init_bulk 仅 VASP MD | M3 |

## 4. 数据约定（全链统一）

| 文件 | 形状 | 含义 | 生成处 |
|---|---|---|---|
| `fparam.npy` | (n,1) | 帧电子温度 K | post_fp（job.json ele_temp → degauss 反推或直接记录） |
| `aparam.npy` | (n,natoms) | 原子温度 K（use_ele_temp=2 语义） | post_fp / AIMD 导出 |
| `uparam.npy` | (n,1) 或 (n,natoms) | Hubbard U eV，frame/atomic 两模式 | HP 链标定输出；DP+U 输入、MLU 标签 |
| `virial.npy` | (n,9) | eV（QE 应力 → W=−σV） | 标定输出 |

- dpdata 对 f/a/uparam 的传递不可靠（uparam 完全不支持）：所有涉及这三个文件的环节用自实现 npy 直读直写
- 磁性标注：`sys_idx`（第几个 sys_configs 体系）→ param 新键 `magnetic_sys_idx`（列表），构型从来源系统继承
- 赝势/ecut/kspacing 与 mlu_model 一致：`Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`、ecutwfc=80 / ecutrho=400、kspacing 由 `compute_kmesh`

## 5. 运行环境（容器内开发，后续迁移计算）

- 统一 venv：`/personal/deepmd-kit/.venv`
  - deepmd-kit fork 3.2.0b1.dev103+g72484546b（**非 editable 快照**，含未提交 MLU 层；改 deepmd-kit 源码后需 `pip install .` 重装）
  - dpgen 0.1.dev50+g0b9acec（editable → `/personal/dpgen`）
  - scikit-learn / scipy / matplotlib / dpdata 1.0.2 / numpy 2.1.3
  - CLI：`dp`、`dpgen` 均在该 venv 的 bin 下
- QE 与 SLURM：远程集群（mlu_model 验证环境，QE 7.4.1，partition nudt96c）；本容器无 QE
- LAMMPS：`/opt/deepmd-kit/bin/lmp`（deepmd 3.1.2，**无 MLU/uparam 关键字**）——带 uparam 的探索需用 deepmd-kit fork 自行编译的 LAMMPS（`.devcontainer/build_cxx_mlu.sh`），迁移平台后构建

## 6. 已知坑（移植 mlu_model 时一并修复）

1. 生成的 SLURM 脚本中 `continue` 出现在无循环上下文（`err_0.log` 实证）→ 改 if 块包裹
2. `HUBBARD` 卡硬编码 `Fe-3d` → 参数化 element/manifold
3. mlu_model README 提到的 collect-u CSV 报告未实现 → 本仓库实现
4. `run_fp_inner` 的 check_fin 对 HP 链要以"链全部完成"为准（scf+hp+scf_new 三个 output 都 JOB DONE）
