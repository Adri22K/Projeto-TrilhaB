# Dicionário de dados v0.2

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

## Alvo

| Variável | Tipo | Definição |
|---|---|---|
| `qualidade_inadequada_24h` | binária (`0`/`1`) | `1` quando, 24 horas após o instante das features, ao menos um poluente atinge a classe CETESB N3 — RUIM ou pior; `0` caso contrário |

O estado N3 é calculado com os limites: PM10 (média de 24 h) > 100 µg/m³; PM2,5 (média de 24 h) > 50 µg/m³; SO2 (média de 24 h) > 50 µg/m³; O3 (média de 8 h) > 130 µg/m³; NO2 (1 h) > 240 µg/m³; CO (média de 8 h) > 11 ppm. Para o CO, usa-se o fator aproximado `ppm = µg/m³ × 0,0008727` a 25 °C e 1 atm.

As colunas auxiliares de construção do alvo (`pm10_ruim`, `pm2_5_ruim`, `so2_ruim`, `ozone_ruim`, `no2_ruim`, `co_ruim`) são excluídas das features.

## Features iniciais

| Padrão | Definição e justificativa |
|---|---|
| `hora`, `dia_semana`, `mes` | Capturam ciclos de atividade humana e sazonalidade sem usar o timestamp como número contínuo |
| `<variável>_lag_1h` | Último valor conhecido; representa persistência de curto prazo |
| `<variável>_lag_24h` | Valor conhecido na mesma hora do dia anterior; representa ciclo diário |
| `<variável>_media_24h` | Média das 24 horas anteriores, sempre após `.shift(1)`; resume o histórico recente sem incluir o instante atual ou o futuro |

Os padrões de lag e média móvel são aplicados aos seis poluentes e às três variáveis meteorológicas. As janelas foram definidas no treino e aplicadas sem alteração ao teste.

## Log de limpeza

O bruto é preservado em `data/raw/`. Tipos são padronizados, sentinelas (`-999`, `-9999`, `9999`) são convertidas em `NaN`, valores negativos em concentrações, precipitação e vento são convertidos em `NaN`, e duplicatas de timestamp são removidas mantendo a primeira ocorrência. As contagens efetivas estão em `reports/sprint2/relatorio_execucao.json`.
