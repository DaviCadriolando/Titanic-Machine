# Titanic Machine Learning - Comparação de Modelos de Classificação

Este repositório contém o projeto completo de Aprendizagem de Máquina desenvolvido em Python no Google Colab para prever a sobrevivência de passageiros do Titanic utilizando a base de dados `train.csv`. O projeto está estruturado com todos os códigos divididos em blocos modulares (KNN, SVM e Random Forest), acompanhados por um relatório técnico completo e a explicação detalhada de todas as métricas de avaliação.

---

## 📂 Estrutura Completa do Código em Blocos

### Bloco 1: Preparação dos Dados, Limpeza e Otimização do KNN

Este código realiza o carregamento da base de dados, a seleção das características (`Pclass`, `Sex`, `Age`), a conversão de variáveis categóricas, o tratamento de valores ausentes pela mediana da idade, a padronização dos dados com o `StandardScaler` e a busca pelo melhor hiperparâmetro $K$ do algoritmo **K-Nearest Neighbors (KNN)** via validação.

```python
# Bloco 1: Importação, Tratamento de Dados e Otimização do KNN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Coletar e carregar o arquivo dos dados
from google.colab import files
files.upload()
dados = pd.read_csv("train.csv")

# Selecionar e transformar características
X = dados[["Pclass", "Sex", "Age"]].copy()
y = dados["Survived"]
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# Separar em treino, validação e teste
X_treino, X_temp, y_treino, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_teste, y_val, y_teste = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Limpar e padronizar os dados
mediana_idade = X_treino["Age"].median()
X_treino = X_treino.fillna({"Age": mediana_idade})
X_val = X_val.fillna({"Age": mediana_idade})
X_teste = X_teste.fillna({"Age": mediana_idade})

escala = StandardScaler()
X_treino_esc = escala.fit_transform(X_treino)
X_val_esc = escala.transform(X_val)
X_teste_esc = escala.transform(X_teste)

# Otimizar o parâmetro K do KNN
valores_k = [1, 3, 5, 7, 11]
resultados_k = []
for k in valores_k:
    candidato = KNeighborsClassifier(n_neighbors=k)
    candidato.fit(X_treino_esc, y_treino)
    previsoes = candidato.predict(X_val_esc)
    resultados_k.append({
        "K": k,
        "Acurácia": accuracy_score(y_val, previsoes),
        "F1-score": f1_score(y_val, previsoes, zero_division=0)
    })

tabela_k = pd.DataFrame(resultados_k)
melhor = tabela_k.sort_values(["F1-score", "Acurácia"], ascending=False).iloc[0]
melhor_k = int(melhor["K"])
print("Melhor K encontrado na validação:", melhor_k)

# Consolidar base de treino final (Treino + Validação)
X_final = X.loc[X_treino.index.union(X_val.index)].copy()
y_final = y.loc[X_final.index]
X_final = X_final.fillna({"Age": X_final["Age"].median()})
X_teste_final = X.loc[X_teste.index].fillna({"Age": X_final["Age"].median()})

escala_final = StandardScaler()
X_final_esc = escala_final.fit_transform(X_final)
X_teste_final_esc = escala_final.transform(X_teste_final)

# Avaliação final do KNN no conjunto de teste
modelo_knn = KNeighborsClassifier(n_neighbors=melhor_k)
modelo_knn.fit(X_final_esc, y_final)
previsoes_knn = modelo_knn.predict(X_teste_final_esc)

acc_knn = accuracy_score(y_teste, previsoes_knn)
prec_knn = precision_score(y_teste, previsoes_knn, zero_division=0)
rec_knn = recall_score(y_teste, previsoes_knn, zero_division=0)
f1_knn = f1_score(y_teste, previsoes_knn, zero_division=0)
```

---

### Bloco 2: Implementação e Avaliação do Support Vector Machine (SVM)

Este código implementa o algoritmo **Support Vector Machine (SVM)**, que busca encontrar o hiperplano ideal no espaço multidimensional para separar os passageiros que sobreviveram daqueles que não sobreviveram, utilizando os dados padronizados.

```python
# Bloco 2: Treinamento e Avaliação do Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Instanciação e treino do SVM
modelo_svm = SVC(random_state=42)
modelo_svm.fit(X_final_esc, y_final)
previsoes_svm = modelo_svm.predict(X_teste_final_esc)

# Cálculo das métricas do SVM
acc_svm = accuracy_score(y_teste, previsoes_svm)
prec_svm = precision_score(y_teste, previsoes_svm, zero_division=0)
rec_svm = recall_score(y_teste, previsoes_svm, zero_division=0)
f1_svm = f1_score(y_teste, previsoes_svm, zero_division=0)
cm_svm = confusion_matrix(y_teste, previsoes_svm)

print("Support Vector Machine (SVM)")
print(f"Acurácia:  {acc_svm:.2%}")
print(f"Precisão:  {prec_svm:.2%}")
print(f"Recall:    {rec_svm:.2%}")
print(f"F1-score:  {f1_svm:.2%}")
print("Matriz de Confusão:\n", cm_svm)
```

---

### Bloco 3: Implementação do Random Forest e Tabela Comparativa Final

Este código implementa o **Random Forest** (um modelo de *ensemble* baseado em árvores de decisão independentes combinadas por votação majoritária) e gera a consolidação final da tabela comparativa com todos os resultados obtidos.

```python
# Bloco 3: Treinamento do Random Forest e Tabela Comparativa Final
from sklearn.ensemble import RandomForestClassifier

# Instanciação e treino do Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_final, y_final)
previsoes_rf = modelo_rf.predict(X_teste_final)

# Cálculo das métricas do Random Forest
acc_rf = accuracy_score(y_teste, previsoes_rf)
prec_rf = precision_score(y_teste, previsoes_rf, zero_division=0)
rec_rf = recall_score(y_teste, previsoes_rf, zero_division=0)
f1_rf = f1_score(y_teste, previsoes_rf, zero_division=0)
cm_rf = confusion_matrix(y_teste, previsoes_rf)

print("Random Forest")
print(f"Acurácia:  {acc_rf:.2%}")
print(f"Precisão:  {prec_rf:.2%}")
print(f"Recall:    {rec_rf:.2%}")
print(f"F1-score:  {f1_rf:.2%}")
print("Matriz de Confusão:\n", cm_rf)

# Tabela Comparativa Final entre os três modelos
resultados_finais = [
    {"Algoritmo": f"KNN (K={melhor_k})", "Acurácia": acc_knn, "Precisão": prec_knn, "Recall": rec_knn, "F1-score": f1_knn},
    {"Algoritmo": "SVM", "Acurácia": acc_svm, "Precisão": prec_svm, "Recall": rec_svm, "F1-score": f1_svm},
    {"Algoritmo": "Random Forest", "Acurácia": acc_rf, "Precisão": prec_rf, "Recall": rec_rf, "F1-score": f1_rf}
]

tabela_final = pd.DataFrame(resultados_finais)
print("\n" + "=" * 60)
print("TABELA COMPARATIVA FINAL")
print("=" * 60)
print(tabela_final.round(4).to_string(index=False))
```

---

## 📊 Relatório Técnico e Explicação das Métricas

### 1. Descrição dos Algoritmos Utilizados

* **K-Nearest Neighbors (KNN):** Classifica novos pontos com base na proximidade geométrica (distância euclidiana) em relação aos seus vizinhos mais próximos no espaço de características.
* **Support Vector Machine (SVM):** Constrói um hiperplano ótimo de separação entre as classes, maximizando a margem geométrica de segurança entre os grupos de sobreviventes e não sobreviventes.
* **Random Forest:** Modelo baseado em uma floresta de árvores de decisão independentes treinadas com subconjuntos aleatórios de dados, combinando as respostas por votação majoritária para reduzir a variância e evitar o *overfitting*.

### 2. Explicação e Cálculo das Métricas de Avaliação

* **Acurácia (Accuracy):** Mede a proporção geral de previsões corretas realizadas pelo modelo em relação ao total de instâncias testadas.
* **Precisão (Precision):** Avalia a exatidão das previsões positivas. Ou seja, de todos os passageiros que o modelo previu que sobreviveram, quantos realmente sobreviveram (evita falsos positivos).
* **Recall (Sensibilidade):** Mede a capacidade do modelo de identificar corretamente todos os casos positivos reais. Responde à pergunta: *De todas as pessoas que sobreviveram na realidade, quantos o algoritmo conseguiu detectar?*

  $$\text{Recall} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos} + \text{Falsos Negativos}}$$

* **F1-score:** É a média harmônica entre a Precisão e o Recall. Fornece uma medida única de equilíbrio para o modelo, garantindo que ele não apresente um bom desempenho em apenas uma das frentes.

  $$\text{F1-Score} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}$$

* **Matriz de Confusão:** Tabela que cruza os resultados reais com as previsões do modelo, dividindo-se em Verdadeiros Negativos, Falsos Positivos, Falsos Negativos e Verdadeiros Positivos.

### 3. Tabela Comparativa de Resultados no Conjunto de Teste

| Algoritmo | Acurácia | Precisão | Recall | F1-score |
| --- | --- | --- | --- | --- |
| **KNN (K=11)** | 79.10% | 72.55% | 72.55% | 72.55% |
| **Support Vector Machine (SVM)** | 74.63% | 68.09% | 62.75% | 65.31% |
| **Random Forest** | **79.85%** | **77.27%** | 66.67% | 71.58% |
