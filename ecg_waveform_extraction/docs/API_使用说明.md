# ECG 流水线调用接口说明

`ecg_waveform_extraction` 打包为可安装的 Python 包,提供三层调用方式:
Python API(推荐)、命令行 `ecg-pipeline`、以及保留的逐阶段 CLI。

- 检测算法全部为库实现(sleepecg / wavedet / prominence-delineator / neurokit2),
  包内代码只做编排、窗采样、门控与判定。
- 判定(阶段 7)不读取任何设备结论字段;设备字段仅作对照列导出。

---

## 1. 安装

```bash
cd ecg_waveform_extraction          # 本仓库根(即包根)
pip install .                       # 装进当前环境
# 或构建 wheel:
pip wheel . -w dist --no-build-isolation
pip install dist/ecg_waveform_extraction-1.0.0-py3-none-any.whl
```

依赖见 `pyproject.toml`(与 `requirements.txt` 一致;Python ≥ 3.10)。
安装后命令行入口 `ecg-pipeline` 可用;不安装也能以
`python -m ecg_waveform_extraction.src.api` 从父目录运行。

---

## 2. Python API

### 2.1 批量:`run_pipeline`

```python
from ecg_waveform_extraction.src.api import run_pipeline

res = run_pipeline(
    data_dir=r'C:\datasets\RA-LA_Reversal\aECG',  # .aECG 文件目录;None=用阶段默认路径
    out_dir=r'C:\ecg_out',                        # 必填,所有阶段输出根目录
    records='1805185J6U,220326WBYY',              # 可选,记录名或文件名,字符串或可迭代
    plots=True,                                    # False 仅关闭阶段7专题图
    stages=None,                                   # 可选,阶段子集,如 ('polarity','overview','reversal')
)
```

**参数**

| 参数 | 类型 | 说明 |
|---|---|---|
| `data_dir` | `str\|None` | .aECG 目录;`None` 用阶段默认(本机数据集路径) |
| `out_dir` | `str\|Path` | **必填**。输出根;自动创建 7 个规范子目录 |
| `records` | `str\|Iterable\|None` | 队列过滤;实现方式是把选中文件复制到 `out_dir/_data_subset`(阶段 1 没有 `--records` 参数,这样 7 个阶段看到同一子集) |
| `plots` | `bool` | `False` = 阶段 7 不出图(阶段 2-4 的逐记录图是各自脚本内建的,不受此开关控制) |
| `stages` | `Iterable[str]\|None` | 子集(自动按规范顺序执行),用于在既有输出上增量重跑 |
| `verbose` | `bool` | 打印每阶段调用的等价命令行 |

**返回 dict**(键存在的前提是对应阶段已跑过):

```python
{'out_dir': 'C:\\ecg_out',
 'stages_run': ['rpeaks', ..., 'reversal'],
 'reversal_csv':        '...\\reversal_reflead\\reversal_by_record.csv',
 'reversal_simple_csv': '...\\reversal_reflead\\reversal_simple.csv',   # 中文速览表
 'overview_csv':        '...\\record_overview\\record_overview.csv',
 'cohort_summary':      '...\\reversal_reflead\\cohort_summary.txt',
 'elapsed_sec': 123.4}
```

### 2.2 单条:`analyze_record`

```python
from ecg_waveform_extraction.src.api import analyze_record

r = analyze_record(r'C:\datasets\...\1805185J6U.aECG')
print(r['verdict'], r['reversal_type'], r['polarity']['I'])
```

把单个文件复制进临时工作区(默认用完即删),跑全链,返回**纯 dict**(可 `json.dumps`)。
`work_dir='C:\x'` 指定并保留工作区(含全部中间 CSV 与图),`keep_workdir=True` 不删临时区。

**返回字段**

| 键 | 含义 |
|---|---|
| `verdict` | `definite / probable / suspicious / none / poor_P` |
| `reversal_type` | `ra_la / ra_ll / la_ll`;`verdict` 为 none/poor_P 时为空串 |
| `score` | 证据分(胜出类型的那套权重) |
| `signatures` | 7 个硬签名布尔:`S1/S2/S3`(RA-LA)、`T1/T2`(RA-LL)、`U1/U2`(LA-LL) |
| `fired` / `notes` | 触发的规则串 / 鉴别诊断注记(dextrocardia?、P_axis_nonsinus 等) |
| `p_frontal_axis_deg` | 我们测的 P 额面轴(导联 I/aVF 中位符号幅度 atan2) |
| `p_amplitude_mv` / `p_pass_rate_pct` | P 幅度 / P 门通过率(<50% 判 poor_P) |
| `hr_median_bpm` | 阶段 6 的心率中位数 |
| `polarity` | `{导联: {P,QRS,T: positive/negative/biphasic/flat}}` × 12 导联(多数极性) |
| `device` | 仅对照:`P_axis_deg / flagged_reversal / interpretation` |

### 2.3 结果回读

```python
from ecg_waveform_extraction.src.api import read_reversal, read_overview

df  = read_reversal(r'C:\ecg_out')                 # 阶段7全表(或 simple=True 拿中文速览表)
ov  = read_overview(r'C:\ecg_out')                 # 阶段6总览
```
两个函数都带 `.new.csv` 兜底(目标 CSV 被 Excel 锁住时阶段会写 `.new.csv`)。

---

## 3. 命令行

```bash
# 批量
ecg-pipeline run --data-dir C:\datasets\aECG --out-dir C:\ecg_out \
                 [--records 1805185J6U,220326WBYY] [--no-plots] \
                 [--stages rpeaks,qrs,twave,pwave,polarity,overview,reversal]

# 单条(人类可读摘要;--json 出全字段)
ecg-pipeline analyze C:\datasets\aECG\1805185J6U.aECG --json

# 已完成运行的判定计数
ecg-pipeline results --out-dir C:\ecg_out
```

---

## 4. 输出目录结构(out_dir 下)

```
out_dir/
├── pan_tompkins_rpeaks_ref/   # 阶段1:r_pk.csv、summary、图
├── qrs_wavedet_reflead/       # 阶段2:qrs_by_beat.csv、summary_by_record.csv
├── twave_wavedet_reflead/     # 阶段3:t_by_beat.csv、summary_by_record.csv
├── pwave_reflead/             # 阶段4:p_by_beat.csv、summary_by_record.csv
├── polarity_reflead/          # 阶段5:polarity_by_beat_lead.csv(逐搏×12导联)
├── record_overview/           # 阶段6:record_overview.csv(666×81)
├── reversal_reflead/          # 阶段7:reversal_by_record.csv、reversal_simple.csv、
│                              #       cohort_summary.txt、plots/*.png
└── _data_subset/              # 仅当 records 过滤时存在(复制的队列文件)
```

---

## 5. 逐阶段 CLI(保留)

每阶段仍是独立模块,可在父目录直接跑(参数与 `run_pipeline` 等价):

```bash
python -m ecg_waveform_extraction.src.batch_rpeaks_pt_reflead --data-dir D --out-dir O
python -m ecg_waveform_extraction.src.batch_qrs_wavedet_reflead ...
python -m ecg_waveform_extraction.src.batch_twave_wavedet_reflead ...
python -m ecg_waveform_extraction.src.batch_pwave_reflead ... --t-stage-dir ...
python -m ecg_waveform_extraction.src.batch_polarity_reflead ... --p-dir --t-dir --q-dir
python -m ecg_waveform_extraction.src.batch_record_overview --qrs-dir --t-dir --p-dir --pol-dir --out-dir
python -m ecg_waveform_extraction.src.batch_reversal_reflead ... --pol-dir --overview --out-dir --no-plots
```

阶段 2/3 在进程内确定性重做参考导联选择与 PT 检测(与阶段 1 输出一致),
因此链上真正的目录依赖只有:4←3、5←2/3/4、6←2/3/4/5、7←5/6。

## 6. 注意事项

- 单条分析全链耗时约几十秒(阶性 5 极性传播占大头);666 条全量约 1 小时量级。
- 输出目录下的 CSV 被占用(Excel 打开)时自动落 `.new.csv` 并告警。
- 判定阈值与鉴别逻辑(RAD 守卫、P 稳定性门等)见
  `src/batch_reversal_reflead.py` 模块 docstring 的 Validation 段。
