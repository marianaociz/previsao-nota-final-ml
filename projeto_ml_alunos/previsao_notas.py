"""
Previsão de Nota Final de Alunos com Machine Learning
Integrantes: Mariana Ocireu e Rebeca Matewanga

Este script:
1. Gera uma base de dados fake de alunos (horas de estudo, frequência,
   atividades entregues, nota anterior -> nota final).
2. Treina um modelo de Regressão Linear para prever a nota final.
3. Calcula métricas de erro: MAE, RMSE, desvio padrão dos erros e
   intervalo de confiança (95%) do erro.
4. Gera gráficos: dispersão real vs. previsto, distribuição dos erros
   (resíduos) e importância das variáveis (coeficientes).
5. Imprime uma interpretação automática dos resultados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

# ----------------------------------------------------------------------
# 1. GERAÇÃO DA BASE DE DADOS FAKE
# ----------------------------------------------------------------------
np.random.seed(42)
N = 300  # número de alunos simulados

horas_estudo = np.clip(np.random.normal(loc=8, scale=3, size=N), 0, 20)
frequencia = np.clip(np.random.normal(loc=85, scale=10, size=N), 40, 100)
atividades_entregues = np.clip(np.random.normal(loc=8, scale=2.5, size=N), 0, 10).round()
nota_anterior = np.clip(np.random.normal(loc=6.5, scale=1.5, size=N), 0, 10)

# Regra "real" (com ruído) que gera a nota final, simulando relação
# plausível entre as variáveis e o desempenho do aluno.
ruido = np.random.normal(loc=0, scale=0.6, size=N)

nota_final = (
    0.20 * horas_estudo
    + 0.04 * frequencia
    + 0.35 * atividades_entregues
    + 0.45 * nota_anterior
    - 2.0
    + ruido
)
nota_final = np.clip(nota_final, 0, 10)

df = pd.DataFrame({
    "horas_estudo": horas_estudo.round(1),
    "frequencia_%": frequencia.round(1),
    "atividades_entregues": atividades_entregues.astype(int),
    "nota_anterior": nota_anterior.round(1),
    "nota_final": nota_final.round(2),
})

df.to_csv("dados_alunos.csv", index=False)
print(f"Base de dados gerada com {len(df)} alunos e salva em dados_alunos.csv")
print(df.head(10))
print()

# ----------------------------------------------------------------------
# 2. TREINAMENTO DO MODELO
# ----------------------------------------------------------------------
X = df[["horas_estudo", "frequencia_%", "atividades_entregues", "nota_anterior"]]
y = df["nota_final"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Coeficientes do modelo (impacto de cada variável na nota final):")
for nome, coef in zip(X.columns, modelo.coef_):
    print(f"  {nome:25s}: {coef:+.4f}")
print(f"  {'intercepto':25s}: {modelo.intercept_:+.4f}")
print()

# ----------------------------------------------------------------------
# 3. CÁLCULOS ESTATÍSTICOS
# ----------------------------------------------------------------------
erros = y_test.values - y_pred  # erro = valor real - valor previsto

media_erro = np.mean(erros)
desvio_padrao_erro = np.std(erros, ddof=1)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Intervalo de confiança de 95% para o erro médio (distribuição t de Student)
n = len(erros)
confianca = 0.95
t_critico = stats.t.ppf((1 + confianca) / 2, df=n - 1)
erro_padrao = desvio_padrao_erro / np.sqrt(n)
margem_erro = t_critico * erro_padrao
ic_inferior = media_erro - margem_erro
ic_superior = media_erro + margem_erro

print("=" * 60)
print("ESTATÍSTICAS DAS PREVISÕES (conjunto de teste)")
print("=" * 60)
print(f"Quantidade de alunos no teste : {n}")
print(f"Média dos erros                : {media_erro:.4f}")
print(f"Desvio padrão dos erros        : {desvio_padrao_erro:.4f}")
print(f"MAE  (Erro Absoluto Médio)     : {mae:.4f}")
print(f"RMSE (Raiz do Erro Quadrático) : {rmse:.4f}")
print(f"R² (coeficiente de determinação): {r2:.4f}")
print(f"Intervalo de confiança 95% do erro médio: "
      f"[{ic_inferior:.4f}, {ic_superior:.4f}]")
print("=" * 60)

# Salva as estatísticas em um arquivo texto para reaproveitar no relatório
with open("estatisticas.txt", "w", encoding="utf-8") as f:
    f.write("ESTATÍSTICAS DAS PREVISÕES (conjunto de teste)\n")
    f.write(f"Quantidade de alunos no teste: {n}\n")
    f.write(f"Média dos erros: {media_erro:.4f}\n")
    f.write(f"Desvio padrão dos erros: {desvio_padrao_erro:.4f}\n")
    f.write(f"MAE (Erro Absoluto Médio): {mae:.4f}\n")
    f.write(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}\n")
    f.write(f"R² (coeficiente de determinação): {r2:.4f}\n")
    f.write(f"Intervalo de confiança 95% do erro médio: "
            f"[{ic_inferior:.4f}, {ic_superior:.4f}]\n")

# ----------------------------------------------------------------------
# 4. GRÁFICOS
# ----------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# --- Gráfico 1: Nota real vs. Nota prevista ---
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="#2563eb", edgecolor="white", s=60)
lims = [0, 10]
plt.plot(lims, lims, "r--", linewidth=2, label="Previsão perfeita")
plt.xlabel("Nota final real")
plt.ylabel("Nota final prevista")
plt.title("Nota Real vs. Nota Prevista pelo Modelo")
plt.legend()
plt.xlim(lims)
plt.ylim(lims)
plt.tight_layout()
plt.savefig("grafico_real_vs_previsto.png", dpi=150)
plt.close()

# --- Gráfico 2: Distribuição dos erros (resíduos) ---
plt.figure(figsize=(7, 6))
plt.hist(erros, bins=15, color="#16a34a", edgecolor="white", alpha=0.85)
plt.axvline(media_erro, color="red", linestyle="--", linewidth=2,
            label=f"Média = {media_erro:.2f}")
plt.axvline(media_erro + desvio_padrao_erro, color="orange", linestyle=":",
            linewidth=2, label=f"+1 desvio padrão")
plt.axvline(media_erro - desvio_padrao_erro, color="orange", linestyle=":",
            linewidth=2)
plt.xlabel("Erro (Real - Previsto)")
plt.ylabel("Frequência")
plt.title("Distribuição dos Erros de Previsão")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_distribuicao_erros.png", dpi=150)
plt.close()

# --- Gráfico 3: Importância das variáveis (coeficientes) ---
plt.figure(figsize=(7, 6))
cores = ["#2563eb" if c > 0 else "#dc2626" for c in modelo.coef_]
plt.barh(X.columns, modelo.coef_, color=cores)
plt.xlabel("Coeficiente (impacto na nota final)")
plt.title("Importância de Cada Variável no Modelo")
plt.tight_layout()
plt.savefig("grafico_importancia_variaveis.png", dpi=150)
plt.close()

print("\nGráficos salvos: grafico_real_vs_previsto.png, "
      "grafico_distribuicao_erros.png, grafico_importancia_variaveis.png")

# ----------------------------------------------------------------------
# 5. INTERPRETAÇÃO AUTOMÁTICA
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("INTERPRETAÇÃO DOS RESULTADOS")
print("=" * 60)
if mae < 0.6 and r2 > 0.7:
    avaliacao = "O modelo apresenta desempenho BOM/ACEITÁVEL."
elif mae < 1.0 and r2 > 0.5:
    avaliacao = "O modelo apresenta desempenho RAZOÁVEL."
else:
    avaliacao = "O modelo apresenta desempenho FRACO, precisa de ajustes."

print(avaliacao)
print(f"Em média, a previsão erra {mae:.2f} pontos (numa escala de 0 a 10),")
print(f"com 95% de confiança de que o erro médio real está entre "
      f"{ic_inferior:.2f} e {ic_superior:.2f}.")
