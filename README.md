# Previsão da Nota Final de Alunos com Machine Learning

Atividade de Machine Learning: geração de uma base de dados fake de alunos e
treinamento de um modelo de Regressão Linear para prever a **nota final**
com base em:

- horas de estudo semanais
- frequência (%)
- atividades entregues
- nota anterior

## Integrantes
- Mariana Ocireu
- Rebeca Matewanga

## Conteúdo do repositório
- `previsao_notas.py` — script completo: gera os dados, treina o modelo,
  calcula as estatísticas (média, desvio padrão, MAE, RMSE, R², intervalo
  de confiança do erro) e gera os gráficos.
- `dados_alunos.csv` — base de dados fake gerada (300 alunos).
- `estatisticas.txt` — resumo das métricas calculadas.
- `grafico_real_vs_previsto.png` — dispersão nota real vs. prevista.
- `grafico_distribuicao_erros.png` — histograma dos erros (resíduos).
- `grafico_importancia_variaveis.png` — coeficientes/importância das variáveis.
- `Relatorio_Previsao_Notas_ML.docx` — relatório final em Word com código,
  gráficos, estatísticas e análise.

## Resultado obtido
- MAE: 0,47 | RMSE: 0,57 | R²: 0,77
- Intervalo de confiança 95% do erro médio: [0,09 ; 0,33]
- Avaliação: desempenho **bom/aceitável** para um modelo linear simples
  treinado sobre dados sintéticos.
