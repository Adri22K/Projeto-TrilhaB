# Dicionário de dados v0.3

## Chave e fontes

| Variável | Tipo/unidade | Fonte | Papel |
|---|---|---|---|
| `timestamp` | data e hora local (`America/Sao_Paulo`) | Ambas as APIs | Chave horária do merge; não entra como número no modelo |
| `pm10` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `pm2_5` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `carbon_monoxide` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `nitrogen_dioxide` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `sulphur_dioxide` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `ozone` | µg/m³ | Open-Meteo Air Quality | Poluente bruto; feature e origem do alvo |
| `temperature_2m` | °C | Open-Meteo Historical Weather | Variável meteorológica bruta |
| `precipitation` | mm | Open-Meteo Historical Weather | Variável meteorológica bruta |
| `wind_speed_10m` | km/h | Open-Meteo Historical Weather | Variável meteorológica bruta |

**Natureza e resolução das fontes:** poluentes = Open-Meteo Air Quality (modelo CAMS Global, ≈ 45 km, passo nativo de 3 h entregue em série horária; **modelado**, não medição de estação). Clima = Open-Meteo Historical Weather (reanálise ERA5/ERA5-Land + ECMWF IFS 9 km; horária; **reanálise**). Período: 06/09/2024 00h a 06/09/2026 23h, 17.544 horas, fuso `America/Sao_Paulo` (UTC−3) nas duas fontes.

## Alvo

| Variável | Tipo | Definição |
|---|---|---|
| `qualidade_inadequada_24h` | binária (`0`/`1`) | `1` quando, 24 horas após o instante das features, ao menos um poluente atinge a classe CETESB N3 — RUIM ou pior; `0` caso contrário |

O estado N3 é calculado com os limites: PM10 (média de 24 h) > 100 µg/m³; PM2,5 (média de 24 h) > 50 µg/m³; SO2 (média de 24 h) > 40 µg/m³; O3 (média de 8 h) > 130 µg/m³; NO2 (1 h) > 240 µg/m³; CO (média de 8 h) > 11 ppm. Para o CO, usa-se o fator aproximado `ppm = µg/m³ × 0,0008727` a 25 °C e 1 atm.

As colunas auxiliares de construção do alvo (`pm10_ruim`, `pm2_5_ruim`, `so2_ruim`, `ozone_ruim`, `no2_ruim`, `co_ruim`) são excluídas das features.

## Features iniciais

| Padrão | Definição e justificativa |
|---|---|
| `hora`, `dia_semana`, `mes` | Capturam ciclos de atividade humana e sazonalidade sem usar o timestamp como número contínuo |
| `<variável>_lag_1h` | Último valor conhecido; representa persistência de curto prazo |
| `<variável>_lag_24h` | Valor conhecido na mesma hora do dia anterior; representa ciclo diário |
| `<variável>_media_24h` | Média das 24 horas anteriores, sempre após `.shift(1)`; resume o histórico recente sem incluir o instante atual ou o futuro |

Os padrões de lag e média móvel são aplicados aos seis poluentes e às três variáveis meteorológicas. As janelas foram definidas no treino e aplicadas sem alteração ao teste.

## Features derivadas da Sprint 4 (v0.3)

Todas usam somente o instante `t` e o passado, exceto `elevacao_solar_alvo`, que é astronômica e conhecida com antecedência. Código: `src/sprint4_features.py`. Teste automático de vazamento em `reports/sprint4/resultado_execucao.json`.

| Feature | Tipo/unidade | Definição | Status no pipeline congelado |
|---|---|---|---|
| `ozone_media_8h` | µg/m³ | média do O₃ nas 8 h terminando em `t` (métrica CETESB do O₃) | **mantida** |
| `ozone_max_8h` | µg/m³ | máximo do O₃ nas últimas 8 h | **mantida** |
| `ozone_8h_max_24h` | µg/m³ | maior média de 8 h de O₃ nas últimas 24 h | descartada na seleção |
| `ozone_8h_max_24h_delta` | µg/m³ | `ozone_8h_max_24h(t) − ozone_8h_max_24h(t−24)` | descartada na seleção |
| `horas_ruins_24h` | horas (0–24) | nº de horas em classe N3+ nas últimas 24 h | descartada na seleção |
| `temperature_max_24h` | °C | máxima de temperatura nas últimas 24 h | **mantida** |
| `temperature_max_24h_delta` | °C | `temperature_max_24h(t) − temperature_max_24h(t−24)` | **mantida** |
| `hora_sin`, `hora_cos` | adimensional (−1 a 1) | `sin/cos(2π·hora/24)` | **mantidas** (substituem `hora`) |
| `mes_sin`, `mes_cos` | adimensional (−1 a 1) | `sin/cos(2π·(mês−1)/12)` | **mantidas** (substituem `mes`) |
| `elevacao_solar_alvo` | graus | altura do sol em São Paulo na hora prevista `t + 24 h` (fórmula da NOAA) | **mantida** |
| `estado_atual` | binária | classe N3+ no instante `t` (mesma regra do alvo, só passado) | usada **só** pelo baseline de persistência |

### Lista congelada para a Sprint 5 (25 colunas)

Famílias completas (valor, `lag_1h`, `lag_24h`, `media_24h`) de `pm10`, `sulphur_dioxide`, `precipitation` e `wind_speed_10m`, mais `ozone_media_8h`, `ozone_max_8h`, `temperature_max_24h`, `temperature_max_24h_delta`, `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos` e `elevacao_solar_alvo`. A lista está em `config/params.yaml` (`sprint4.features_congeladas`) e em `models/features_s4.json`.

**Retiradas na seleção (validação temporal dentro do treino):** famílias de `ozone`, `pm2_5`, `nitrogen_dioxide`, `carbon_monoxide` e `temperature_2m`; `hora`, `dia_semana` e `mes` brutos; `ozone_8h_max_24h`, `ozone_8h_max_24h_delta` e `horas_ruins_24h`.

**Preparação congelada:** `SimpleImputer(median)` + `PowerTransformer(yeo-johnson, standardize=True)` dentro do `ColumnTransformer`, ajustado só no treino (`models/column_transformer_s4.joblib`).

## Log de limpeza

O bruto é preservado em `data/raw/`. Tipos são padronizados, sentinelas (`-999`, `-9999`, `9999`) são convertidas em `NaN`, valores negativos em concentrações, precipitação e vento são convertidos em `NaN`, e duplicatas de timestamp são removidas mantendo a primeira ocorrência. As contagens efetivas estão em `reports/sprint2/relatorio_execucao.json`.
