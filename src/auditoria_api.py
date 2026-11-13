t_alvo = t0 + pd.Timedelta(hours=24)

    prevs = {}
    for nome_modelo, pipeline in modelos.items():
        features_modelo = _features_esperadas(pipeline, linha_t0)
        X = linha_t0[features_modelo]
        limiar = THRESHOLDS_MODELOS.get(nome_modelo, 0.50)
        prob = float(pipeline.predict_proba(X)[0, 1])
        pred = int(prob >= limiar)
        prevs[nome_modelo] = {
            "prob": prob,
            "pred": pred,
            "limiar": limiar,
            "rotulo": "🚨 Inadequado (N3+)" if pred == 1 else "🟢 Adequado (N1-N2)",
        }

    # Persistencia
    pred_pers = int(row["estado_atual"])
    prevs["Persistência Temporal"] = {
        "prob": float(pred_pers),
        "pred": pred_pers,
        "limiar": 0.50,
        "rotulo": "🚨 Inadequado (N3+)" if pred_pers == 1 else "🟢 Adequado (N1-N2)",
    }

    real_valor = None
    real_rotulo = None
    poluente_critico = None
    if tem_gt and pd.notna(row["qualidade_inadequada_24h"]):
        real_valor = int(row["qualidade_inadequada_24h"])
        real_rotulo = "🚨 Inadequado (N3+)" if real_valor == 1 else "🟢 Adequado (N1-N2)"
        poluente_critico = row["poluente_critico_24h"]

    return {
        "t0": t0,
        "t_alvo_24h": t_alvo,
        "condicao_atual_t0": "🚨 Inadequado (N3+)" if row["estado_atual"] == 1 else "🟢 Adequado (N1-N2)",
        "poluente_agora": row["poluente_critico_agora"],
        "temp_t0": row["temperature_2m"],
        "vento_t0": row["wind_speed_10m"],
        "o3_t0": row["ozone"],
        "pm25_t0": row["pm2_5"],
        "previsoes": prevs,
        "tem_ground_truth": tem_gt,
        "real_valor": real_valor,
        "real_rotulo": real_rotulo,
        "poluente_critico_real": poluente_critico,
    }


def auditar_janela_24h_seguinte(
    df_janela: pd.DataFrame,
    modelos: dict[str, Any],
    tem_ground_truth: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gera tabela hora a hora para as 24 horas seguintes a t0 com predicoes e auditoria."""
    audit_df = pd.DataFrame()
    audit_df["Data/Hora"] = df_janela["timestamp"].dt.strftime("%d/%m %H:00")
    audit_df["timestamp"] = df_janela["timestamp"]
    audit_df["Temp (°C)"] = df_janela["temperature_2m"].round(1)
    audit_df["Vento (m/s)"] = df_janela["wind_speed_10m"].round(1)
    audit_df["O₃ (ug/m³)"] = df_janela["ozone"].round(1)
    audit_df["PM2.5 (ug/m³)"] = df_janela["pm2_5"].round(1)

    # Previsoes dos modelos geradas em cada hora; cada pipeline recebe seus atributos.
    X_por_modelo = {
        nome: df_janela[_features_esperadas(pipeline, df_janela)]
        for nome, pipeline in modelos.items()
    }
    for nome_modelo, pipeline in modelos.items():
        X = X_por_modelo[nome_modelo]
        limiar = THRESHOLDS_MODELOS.get(nome_modelo, 0.50)
        probs = pipeline.predict_proba(X)[:, 1]
        preds = (probs >= limiar).astype(int)
        audit_df[f"{nome_modelo} (Risco %)"] = (probs * 100).round(1).astype(str) + "%"
        audit_df[f"{nome_modelo} (Previsão)"] = np.where(preds == 1, "Inadequado", "Adequado")

    resumo = {}
    if tem_ground_truth:
        y_real = df_janela["estado_atual"].astype(int).values
        audit_df["Realidade CETESB"] = np.where(y_real == 1, "🚨 Inadequado (N3+)", "🟢 Adequado (N1-N2)")
        audit_df["Poluente Crítico"] = df_janela["poluente_critico_agora"].values

        for nome_modelo, pipeline in modelos.items():
            X = X_por_modelo[nome_modelo]
            limiar = THRESHOLDS_MODELOS.get(nome_modelo, 0.50)
            preds = (pipeline.predict_proba(X)[:, 1] >= limiar).astype(int)

            status_col = []
            for y_t, y_p in zip(y_real, preds):
                if y_t == 1 and y_p == 1:
                    status_col.append("✅ Acerto (Verdadeiro Positivo 🚨)")
                elif y_t == 0 and y_p == 0:
                    status_col.append("✅ Acerto (Verdadeiro Negativo 🟢)")
                elif y_t == 1 and y_p == 0:
                    status_col.append("❌ Falso Negativo (Crítico)")
                else:
                    status_col.append("⚠️ Falso Positivo (Conservador)")

            audit_df[f"Auditoria {nome_modelo}"] = status_col

            acertos = int((preds == y_real).sum())
            total = len(y_real)
            resumo[nome_modelo] = {
                "acertos": acertos,
                "total": total,
                "acuracia_pct": round(acertos / total * 100, 1) if total > 0 else 0,
                "tp": int(((y_real == 1) & (preds == 1)).sum()),
                "tn": int(((y_real == 0) & (preds == 0)).sum()),
                "fp": int(((y_real == 0) & (preds == 1)).sum()),
                "fn": int(((y_real == 1) & (preds == 0)).sum()),
            }

    return audit_df, resumo
