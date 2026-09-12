"""
Funções de treino dos modelos supervisionados (Fase 3).
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score


def treinar_regressao_logistica(X_train, y_train, max_iter=1000):
    """
    Treina uma Regressão Logística (baseline linear).

    Args:
        X_train (pandas.DataFrame): Features de treino.
        y_train (pandas.Series): Alvo de treino.
        max_iter (int): Número máximo de iterações (aumentado do padrão
            de 100 para evitar aviso de não-convergência em bases grandes).

    Returns:
        sklearn.linear_model.LogisticRegression: Modelo treinado.
    """
    modelo = LogisticRegression(max_iter=max_iter)
    modelo.fit(X_train, y_train)
    return modelo


def treinar_arvore_decisao(X_train, y_train, max_depth=None, random_state=42):
    """
    Treina uma Árvore de Decisão.

    Args:
        X_train (pandas.DataFrame): Features de treino.
        y_train (pandas.Series): Alvo de treino.
        max_depth (int, opcional): Profundidade máxima. `None` = sem limite.
        random_state (int): Semente de aleatoriedade, para reprodutibilidade.

    Returns:
        sklearn.tree.DecisionTreeClassifier: Modelo treinado.
    """
    modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    modelo.fit(X_train, y_train)
    return modelo


def treinar_random_forest(X_train, y_train, class_weight="balanced",
                           n_estimators=100, max_depth=None, random_state=42, **kwargs):
    """
    Treina uma Random Forest.

    `class_weight="balanced"` por padrão: para esta aplicação (priorização
    de risco educacional), o custo de não identificar um aluno em risco
    (falso negativo) é maior que o custo de um falso alarme - por isso
    priorizamos recall da classe minoritária sobre acurácia geral.

    Args:
        X_train (pandas.DataFrame): Features de treino.
        y_train (pandas.Series): Alvo de treino.
        class_weight (str ou dict): Ver documentação do scikit-learn.
        n_estimators (int): Número de árvores.
        max_depth (int, opcional): Profundidade máxima de cada árvore.
        random_state (int): Semente de aleatoriedade.
        **kwargs: Outros parâmetros repassados ao RandomForestClassifier
            (ex.: min_samples_split, min_samples_leaf, max_features).

    Returns:
        sklearn.ensemble.RandomForestClassifier: Modelo treinado.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight=class_weight, random_state=random_state,
        n_jobs=-1, **kwargs,
    )
    modelo.fit(X_train, y_train)
    return modelo


def buscar_hiperparametros_random_forest(X_train, y_train, tamanho_amostra=200000,
                                          n_iter=10, cv=3, random_state=42):
    """
    Busca os melhores hiperparâmetros da Random Forest via
    RandomizedSearchCV, otimizando o recall da classe "Não" (0) -
    a classe de risco educacional que queremos priorizar identificar.

    Por viabilidade computacional em bases grandes, a busca é feita numa
    amostra do treino (não a base inteira) - o modelo final deve ser
    retreinado na base completa com os parâmetros encontrados.

    Args:
        X_train (pandas.DataFrame): Features de treino completas.
        y_train (pandas.Series): Alvo de treino completo.
        tamanho_amostra (int): Tamanho da amostra usada na busca.
        n_iter (int): Número de combinações de parâmetros testadas.
        cv (int): Número de folds da validação cruzada.
        random_state (int): Semente de aleatoriedade.

    Returns:
        sklearn.model_selection.RandomizedSearchCV: Objeto de busca já
            ajustado (`.best_params_` e `.best_score_` disponíveis).
    """
    amostra_x = X_train.sample(min(tamanho_amostra, len(X_train)), random_state=random_state)
    amostra_y = y_train.loc[amostra_x.index]

    scorer_recall_nao = make_scorer(recall_score, pos_label=0)

    parametros = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    busca = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=random_state),
        param_distributions=parametros,
        n_iter=n_iter, cv=cv, scoring=scorer_recall_nao,
        random_state=random_state, n_jobs=-1,
    )
    busca.fit(amostra_x, amostra_y)
    return busca


def comparar_modelos(modelos, X_train, y_train, X_test, y_test):
    """
    Treina e compara múltiplos modelos já instanciados, retornando uma
    tabela com acurácia de treino e teste de cada um.

    Args:
        modelos (dict[str, estimator]): Nome do modelo -> instância
            (não treinada) do scikit-learn.
        X_train, y_train, X_test, y_test: Conjuntos de treino/teste.

    Returns:
        pandas.DataFrame: Colunas `modelo`, `acuracia_treino`, `acuracia_teste`.
    """
    resultados = []
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        acc_treino = accuracy_score(y_train, modelo.predict(X_train))
        acc_teste = accuracy_score(y_test, modelo.predict(X_test))
        resultados.append({"modelo": nome, "acuracia_treino": acc_treino, "acuracia_teste": acc_teste})
    return pd.DataFrame(resultados)
