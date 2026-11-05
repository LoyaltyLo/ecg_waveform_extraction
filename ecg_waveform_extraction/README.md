# ecg_waveform_extraction

12 导联 aECG(HL7 XML,10 s,fs=1000 Hz)的波分割与肢体导联反接检测流水线。
检测算法全部为库实现(sleepecg / wavedet / prominence-delineator / neurokit2),
本仓库只做编排、门控与判定逻辑。

## 七阶段链

| # | 阶段 | 模块 | 输出子目录 |
|---|------|------|-----------|
| 1 | R 波(SQI 选参考导联 + Pan-Tompkins) | `src.batch_rpeaks_pt_reflead` | `pan_tompkins_rpeaks_ref/` |
| 2 | QRS 边界(wavedet) | `src.batch_qrs_wavedet_reflead` | `qrs_wavedet_reflead/` |
| 3 | T 波(窗内 wavedet) | `src.batch_twave_wavedet_reflead` | `twave_wavedet_reflead/` |
| 4 | P 波(双库池化 + SNR 门) | `src.batch_pwave_reflead` | `pwave_reflead/` |
| 5 | 12 导联极性 | `src.batch_polarity_reflead` | `polarity_reflead/` |
| 6 | 记录级总览 | `src.batch_record_overview` | `record_overview/` |
| 7 | 肢体反接判定(RA-LA / RA-LL / LA-LL) | `src.batch_reversal_reflead` | `reversal_reflead/` |

## 调用接口

```python
from ecg_waveform_extraction.src.api import run_pipeline, analyze_record

# 批量:一个 .aECG 目录 -> 判定 CSV + 专题图
res = run_pipeline(data_dir=r'C:\...\aECG', out_dir=r'C:\...\out')

# 单条:一个文件 -> 一个 dict(临时工作区,自动清理)
r = analyze_record(r'C:\...\aECG\1805185J6U.aECG')
r['verdict'], r['reversal_type'], r['polarity']['I']
```

命令行(安装后 `ecg-pipeline`,或 `python -m ecg_waveform_extraction.src.api`):

```bash
ecg-pipeline run    --data-dir DIR --out-dir DIR [--records a,b] [--no-plots] [--stages ...]
ecg-pipeline analyze RECORD.aECG [--work-dir DIR] [--keep] [--json]
ecg-pipeline results --out-dir DIR
```

完整参数与返回值字段说明见 [docs/API_使用说明.md](docs/API_使用说明.md)。

## 安装

```bash
pip install .            # 或 pip wheel . -w dist 后安装 wheel
pytest tests             # 回归测试
```
