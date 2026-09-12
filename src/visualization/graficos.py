"""
Funções de visualização (Fase 3).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_matriz_confusao(matriz, titulo="Matriz de Confusão", ax=None):
    """
    Plota uma matriz de confusão como heatmap, com rótulos legíveis
    ("Não"/"Sim") em vez de índices numéricos.

    Args:
        matriz (numpy.ndarray): Matriz de confusão 2x2 (ex.: saída de
            `sklearn.metrics.confusion_matrix`).
        titulo (str): Título do gráfico.
        ax (matplotlib.axes.Axes, opcional): Eixo onde plotar (para
            compor múltiplas matrizes lado a lado). Cria uma figura
            nova se não informado.

    Returns:
        matplotlib.axes.Axes: O eixo usado no plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        matriz, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
        xticklabels=["Não (previsto)", "Sim (previsto)"],
        yticklabels=["Não (real)", "Sim (real)"],
    )
    ax.set_title(titulo)
    ax.set_ylabel("Valor real")
    ax.set_xlabel("Valor previsto")
    return ax


def plot_importancia_features(modelo, colunas, top_n=15, titulo="Features mais importantes"):
    """
    Plota um gráfico de barras horizontais com as `top_n` features mais
    importantes de um modelo (usa `.feature_importances_`, disponível
    em modelos de árvore/floresta do scikit-learn).

    Args:
        modelo: Modelo treinado com atributo `feature_importances_`.
        colunas (list[str] ou pandas.Index): Nomes das features, na
            mesma ordem usada no treino.
        top_n (int): Quantas features mostrar.
        titulo (str): Título do gráfico.

    Returns:
        pandas.Series: Ranking completo de importância (não só o top_n exibido).
    """
    importancias = pd.Series(modelo.feature_importances_, index=colunas).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    importancias.head(top_n).sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Importância")
    ax.set_title(titulo)
    plt.tight_layout()

    return importancias


def plot_arvore_decisao(modelo, feature_names, max_depth_exibicao=3, titulo="Árvore de Decisão"):
    """
    Plota uma árvore de decisão treinada, limitando a profundidade
    exibida na imagem (a árvore treinada pode ter mais níveis - o
    limite aqui é só visual, para manter a imagem legível).

    Args:
        modelo (sklearn.tree.DecisionTreeClassifier): Árvore já treinada.
        feature_names (list[str]): Nomes das features usadas no treino.
        max_depth_exibicao (int): Profundidade máxima mostrada na imagem.
        titulo (str): Título do gráfico.
    """
    from sklearn.tree import plot_tree

    plt.figure(figsize=(26, 14))
    plot_tree(
        modelo, feature_names=feature_names, class_names=["Não", "Sim"],
        filled=True, rounded=True, fontsize=8, max_depth=max_depth_exibicao,
    )
    plt.title(titulo)


def plot_shap_summary(modelo, X_amostra, classe_positiva_idx=1):
    """
    Plota o SHAP summary plot para um modelo de árvore/floresta,
    mostrando o impacto individual de cada feature na previsão de cada
    linha da amostra.

    Args:
        modelo: Modelo de árvore/floresta já treinado (compatível com
            `shap.TreeExplainer`).
        X_amostra (pandas.DataFrame): Amostra do conjunto de teste (uso
            recomendado: algumas centenas de linhas, para manter o
            cálculo rápido).
        classe_positiva_idx (int): Índice da classe positiva nos valores
            SHAP (1 = "Sim", seguindo o LabelEncoder usado no projeto).
    """
    import shap

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_amostra)

    valores_classe = shap_values[:, :, classe_positiva_idx]
    shap.summary_plot(valores_classe, X_amostra)
