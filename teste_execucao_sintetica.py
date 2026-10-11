"""Teste local de integracao COM DADOS SINTETICOS, NAO sao metricas do projeto."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml

BASE = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
with TemporaryDirectory(prefix="sprint3_test_") as td:
    root=Path(td)
    for sub in ["src", "data/interim", "data/processed", "config"]:
        (root/sub).mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(BASE/"src/sprint3_baselines.py", root/"src/sprint3_baselines.py")
    tempo=pd.date_range("2026-03-15", "2026-05-01", freq="h", inclusive="left")
    n=len(tempo)
    hour=np.arange(n)
    spike=((hour//24)%6 < 2).astype(float)
    air=pd.DataFrame({
       "timestamp":tempo,
       "pm10":16+85*spike+RNG.random(n)*4,
       "pm2_5":10+60*spike+RNG.random(n)*3,
       "carbon_monoxide":300+20*spike,
       "nitrogen_dioxide":30+4*spike,
       "sulphur_dioxide":10+4*spike,
       "ozone":50+10*spike,
       "temperature_2m":25+np.sin(hour/24),
       "precipitation":np.maximum(0,np.sin(hour/60)),
       "wind_speed_10m":4+np.cos(hour/10),
    })
    air.to_csv(root/"data/interim/integrado_limpo.csv",index=False)
    bad=((air.pm10.rolling(24,min_periods=24).mean()>100) |
         (air.pm2_5.rolling(24,min_periods=24).mean()>50) |
         (air.ozone.rolling(8,min_periods=8).mean()>130)).astype('Int64')
    air['qualidade_inadequada_24h'] = bad.shift(-24)
    air['hora']=air.timestamp.dt.hour
    air['dia_semana']=air.timestamp.dt.dayofweek
    air['mes']=air.timestamp.dt.month
    for col in ['pm10','pm2_5','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone','temperature_2m','precipitation','wind_speed_10m']:
        air[f'{col}_lag_1h']=air[col].shift(1)
        air[f'{col}_lag_24h']=air[col].shift(24)
        air[f'{col}_media_24h']=air[col].shift(1).rolling(24,min_periods=24).mean()
    cut=pd.Timestamp("2026-04-14")
    for name, frame in [('treino',air.loc[air.timestamp<cut]),('teste',air.loc[air.timestamp>=cut])]:
        frame.dropna().to_csv(root/f'data/processed/{name}_features.csv',index=False)
    params={"sprint2":{"forecast_horizon_hours":24,"target":{"name":"qualidade_inadequada_24h",
      "thresholds":{"pm10_24h_ug_m3":100,"pm2_5_24h_ug_m3":50,"sulphur_dioxide_24h_ug_m3":50,"ozone_8h_ug_m3":130,"nitrogen_dioxide_1h_ug_m3":240,"carbon_monoxide_8h_ppm":11},
      "co_ug_m3_to_ppm_factor":0.0008727}}}
    (root/'config/params.yaml').write_text(yaml.safe_dump(params),encoding='utf-8')
    result=subprocess.run([sys.executable,'src/sprint3_baselines.py'],cwd=root,text=True,capture_output=True)
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f'Falha em teste com dados sinteticos: {result.returncode}')
    produced=list((root/'reports/sprint3').glob('*'))
    print('TESTE SINTETICO OK:',len(produced),'arquivos gerados')
    assert len(produced)==9, [p.name for p in produced]
    cmp=pd.read_csv(root/'reports/sprint3/comparativo_baselines.csv')
    assert len(cmp)==3
    assert (cmp['tn']+cmp['fp']+cmp['fn']+cmp['tp']).eq(cmp['n_teste']).all()
