"""
Funções de pré-processamento da base de modelagem.

Consolida, em funções reutilizáveis, as decisões de limpeza tomadas
durante a análise exploratória
"""

import pandas as pd


COLUNAS_LEAKAGE = [
    "proficiencia",
    "taxa_alfabetizacao_municipio",
    "taxa_alfabetizacao_uf",
    "media_portugues_municipio",
    "media_portugues_uf",
]

COLUNAS_MULTICOLINEARIDADE = [
    "pib",
    "quantidade_escolas_inse",
    "taxa_reprovacao_ef_2_ano_anterior",
    "icg_nivel_1", "icg_nivel_2", "icg_nivel_3",
    "icg_nivel_4", "icg_nivel_5", "icg_nivel_6",
    "meta_alfabetizacao_2024_brasil",
]

COLUNAS_SINAL_FRACO = ["had_ef_anos_iniciais", "peso_aluno"]

COLUNAS_NAO_FEATURE = [
    "id_aluno", "id_escola", "id_municipio", "id_municipio_nome",
    "ano", "serie", "presenca", "preenchimento_caderno",
    "sigla_uf_nome", "_gold_processed_at",
]

COLUNAS_NUMERICAS_PARA_PADRONIZAR = [
    "populacao", "pib_per_capita", "atu_ef_anos_iniciais", "dsu_ef_anos_iniciais",
    "afd_ef_anos_iniciais_grupo_1", "ird_alta", "ird_baixa_regularidade",
    "tdi_ef_2_ano_anterior", "taxa_aprovacao_ef_2_ano_anterior",
    "taxa_abandono_ef_2_ano_anterior", "inse_medio",
    "meta_alfabetizacao_2024_municipio", "meta_alfabetizacao_2024_uf",
]


def filtrar_escopo_modelagem(df):
    """
    Aplica o filtro de escopo da modelagem: mantém só alunos presentes
    na avaliação.

    O rótulo `alfabetizado = Não` de um aluno ausente é uma convenção
    do sistema de avaliação (sem nota registrada), não uma medição real
    da alfabetização - por isso esses alunos são excluídos do escopo do
    modelo supervisionado.

    Args:
        df (pandas.DataFrame): Base bruta (ex.: alunos_alfabetizacao.parquet).

    Returns:
        pandas.DataFrame: Só os alunos com `presenca == "Presente"`.
    """
    return df[df["presenca"] == "Presente"].copy()


def remover_colunas_leakage(df):
    """
    Remove as colunas identificadas como data leakage.

    `proficiencia` define o alvo diretamente (corte de 743 pontos).
    As demais são agregados de município/UF calculados incluindo o
    próprio aluno na conta - "vazam" o resultado de forma indireta.

    Args:
        df (pandas.DataFrame): Base já filtrada por escopo.

    Returns:
        pandas.DataFrame: Sem as colunas de leakage.
    """
    colunas_presentes = [c for c in COLUNAS_LEAKAGE if c in df.columns]
    return df.drop(columns=colunas_presentes)


def remover_multicolinearidade(df):
    """
    Remove colunas redundantes identificadas na matriz de correlação
    (população/PIB/quantidade de escolas do INSE; taxa de
    aprovação/reprovação; os 6 níveis do indicador de complexidade de
    gestão escolar; meta nacional, que tem variância zero).

    Args:
        df (pandas.DataFrame): Base sem leakage.

    Returns:
        pandas.DataFrame: Sem as colunas redundantes.
    """
    colunas_presentes = [c for c in COLUNAS_MULTICOLINEARIDADE if c in df.columns]
    return df.drop(columns=colunas_presentes)


def tratar_meta_municipio_ausente(df):
    """
    Trata o nulo estrutural de `meta_alfabetizacao_2024_municipio`
    (ausente para 100% da rede Estadual, por não se aplicar a esse tipo
    de rede - a rede Estadual é regida pela meta estadual).

    Cria uma coluna indicadora de disponibilidade e preenche o nulo com
    a mediana, em vez de descartar as linhas (o que eliminaria ~10% da
    base).

    Args:
        df (pandas.DataFrame): Base com a coluna `meta_alfabetizacao_2024_municipio`.

    Returns:
        pandas.DataFrame: Com a coluna preenchida e a indicadora adicionada.
    """
    df = df.copy()
    if "meta_alfabetizacao_2024_municipio" not in df.columns:
        return df

    df["meta_alfabetizacao_2024_municipio_disponivel"] = (
        df["meta_alfabetizacao_2024_municipio"].notna().astype(int)
    )
    mediana = df["meta_alfabetizacao_2024_municipio"].median()
    df["meta_alfabetizacao_2024_municipio"] = df["meta_alfabetizacao_2024_municipio"].fillna(mediana)
    return df


def preparar_features(df):
    """
    Pipeline completa de preparação de features: remove colunas de
    sinal fraco e não-feature, remove nulos residuais, codifica
    categóricas (one-hot, drop_first=True) e o alvo (LabelEncoder).

    Args:
        df (pandas.DataFrame): Base após remover leakage e multicolinearidade,
            com `meta_alfabetizacao_2024_municipio` já tratada.

    Returns:
        tuple(pandas.DataFrame, pandas.Series, sklearn.preprocessing.LabelEncoder):
            X (features), y (alvo codificado) e o encoder usado (para
            `inverse_transform` depois).
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()

    colunas_sinal_fraco_presentes = [c for c in COLUNAS_SINAL_FRACO if c in df.columns]
    colunas_nao_feature_presentes = [c for c in COLUNAS_NAO_FEATURE if c in df.columns]
    df = df.drop(columns=colunas_sinal_fraco_presentes + colunas_nao_feature_presentes)

    colunas_residuais = [
        "dsu_ef_anos_iniciais", "afd_ef_anos_iniciais_grupo_1", "ird_alta",
        "ird_baixa_regularidade", "tdi_ef_2_ano_anterior",
        "taxa_aprovacao_ef_2_ano_anterior", "taxa_abandono_ef_2_ano_anterior",
        "inse_medio",
    ]
    colunas_residuais_presentes = [c for c in colunas_residuais if c in df.columns]
    df = df.dropna(subset=colunas_residuais_presentes)

    df = pd.get_dummies(df, columns=["rede"], drop_first=True)
    df = pd.get_dummies(df, columns=["sigla_uf"], drop_first=True)

    le = LabelEncoder()
    df["alfabetizado"] = le.fit_transform(df["alfabetizado"])

    y = df["alfabetizado"]
    X = df.drop(columns=["alfabetizado"])

    return X, y, le


def padronizar_features_numericas(X_train, X_test, colunas=None):
    """
    Padroniza (StandardScaler) as colunas numéricas, ajustando (`fit`)
    exclusivamente no conjunto de treino - evita vazamento de
    informação do teste para o treino.

    Args:
        X_train (pandas.DataFrame): Conjunto de treino.
        X_test (pandas.DataFrame): Conjunto de teste.
        colunas (list[str], opcional): Colunas a padronizar. Usa
            `COLUNAS_NUMERICAS_PARA_PADRONIZAR` se não informado.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame, sklearn.preprocessing.StandardScaler):
            X_train e X_test com as colunas padronizadas, e o scaler ajustado.
    """
    from sklearn.preprocessing import StandardScaler

    colunas = colunas or [c for c in COLUNAS_NUMERICAS_PARA_PADRONIZAR if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    scaler.fit(X_train[colunas])
    X_train[colunas] = scaler.transform(X_train[colunas])
    X_test[colunas] = scaler.transform(X_test[colunas])

    return X_train, X_test, scaler
