"""
Funções de avaliação de modelos (Fase 3).
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def avaliar_modelo(modelo, X_test, y_test, nomes_classes=("Não", "Sim")):
    """
    Avalia um modelo já treinado no conjunto de teste, retornando
    acurácia, relatório de classificação (precisão/recall/F1 por
    classe) e matriz de confusão.

    Acurácia isolada pode esconder desempenho ruim numa classe
    minoritária - por isso o relatório completo é sempre retornado
    junto, não só a acurácia.

    Args:
        modelo: Modelo já treinado do scikit-learn (precisa ter `.predict`).
        X_test (pandas.DataFrame): Features de teste.
        y_test (pandas.Series): Alvo real de teste.
        nomes_classes (tuple[str, str]): Nomes das classes (0, 1) para exibição.

    Returns:
        dict: Com chaves `acuracia`, `relatorio` (string) e `matriz_confusao`
            (numpy.ndarray).
    """
    y_pred = modelo.predict(X_test)

    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "relatorio": classification_report(y_test, y_pred, target_names=list(nomes_classes)),
        "matriz_confusao": confusion_matrix(y_test, y_pred),
    }


def comparar_recall_com_sem_balanceamento(modelo_sem_balanceamento, modelo_com_balanceamento,
                                            X_test, y_test):
    """
    Compara diretamente o recall de cada classe entre um modelo sem
    balanceamento de classe e um com `class_weight="balanced"` - usado
    para justificar a escolha do modelo final baseada no trade-off
    entre acurácia geral e recall da classe de risco.

    Args:
        modelo_sem_balanceamento: Modelo treinado sem `class_weight`.
        modelo_com_balanceamento: Modelo treinado com `class_weight="balanced"`.
        X_test (pandas.DataFrame): Features de teste.
        y_test (pandas.Series): Alvo real de teste.

    Returns:
        pandas.DataFrame: Comparação lado a lado das métricas principais.
    """
    resultado_sem = avaliar_modelo(modelo_sem_balanceamento, X_test, y_test)
    resultado_com = avaliar_modelo(modelo_com_balanceamento, X_test, y_test)

    from sklearn.metrics import recall_score
    y_pred_sem = modelo_sem_balanceamento.predict(X_test)
    y_pred_com = modelo_com_balanceamento.predict(X_test)

    tabela = pd.DataFrame({
        "metrica": ["recall_nao", "recall_sim", "acuracia"],
        "sem_balanceamento": [
            recall_score(y_test, y_pred_sem, pos_label=0),
            recall_score(y_test, y_pred_sem, pos_label=1),
            resultado_sem["acuracia"],
        ],
        "com_balanceamento": [
            recall_score(y_test, y_pred_com, pos_label=0),
            recall_score(y_test, y_pred_com, pos_label=1),
            resultado_com["acuracia"],
        ],
    })
    return tabela
