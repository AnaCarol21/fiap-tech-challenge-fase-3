# Dicionário de Dados — Base de Modelagem (Fase 3)

Este dicionário documenta as colunas finais da base usada para treinar os modelos (`X`/`y`), depois de todo o tratamento descrito no README (filtro de escopo, remoção de leakage, tratamento de multicolinearidade e nulos, feature engineering e encoding).

Total: **37 features + 1 alvo**, **1.501.374 registros** (ano-base 2023).

---

## Alvo

| Coluna | Tipo | Descrição | Codificação |
|---|---|---|---|
| `alfabetizado` | Binário | Se o aluno foi considerado alfabetizado na avaliação | `LabelEncoder`: `Não` = 0, `Sim` = 1 |

---

## Features numéricas (13) — todas padronizadas com `StandardScaler`

| Coluna | Descrição | Fonte | Observação |
|---|---|---|---|
| `populacao` | População estimada do município | IBGE (População) | — |
| `pib_per_capita` | PIB per capita do município (`pib / populacao`) | IBGE (PIB dos Municípios) | Feature derivada — evita confundir tamanho do município com riqueza |
| `atu_ef_anos_iniciais` | Alunos por turma, anos iniciais do Ensino Fundamental | INEP (Indicadores Educacionais) | — |
| `dsu_ef_anos_iniciais` | % de docentes com curso superior, anos iniciais | INEP (Indicadores Educacionais) | Correlação negativa com o alvo (ver insight no README) |
| `afd_ef_anos_iniciais_grupo_1` | Adequação da formação docente (grupo mais adequado) | INEP (Indicadores Educacionais) | — |
| `ird_alta` | % de docentes com alta regularidade (baixa rotatividade) | INEP (Indicadores Educacionais) | — |
| `ird_baixa_regularidade` | % de docentes com baixa regularidade (alta rotatividade) | INEP (Indicadores Educacionais) | — |
| `tdi_ef_2_ano_anterior` | Distorção idade-série, 2º ano, **ano anterior (2022)** | INEP (Indicadores Educacionais) | Defasada 1 ano para evitar data leakage |
| `taxa_aprovacao_ef_2_ano_anterior` | Taxa de aprovação, 2º ano, **ano anterior (2022)** | INEP (Indicadores Educacionais) | Defasada 1 ano para evitar data leakage |
| `taxa_abandono_ef_2_ano_anterior` | Taxa de abandono, 2º ano, **ano anterior (2022)** | INEP (Indicadores Educacionais) | Defasada 1 ano para evitar data leakage |
| `inse_medio` | Nível socioeconômico médio das escolas do município | INEP (INSE) | Agregado de escola para município (ver limitações no README) |
| `meta_alfabetizacao_2024_municipio` | Meta de alfabetização do município para 2024 | INEP (Meta de Alfabetização Municipal) | Nulos (rede Estadual + resíduo) preenchidos com mediana — ver coluna indicadora abaixo |
| `meta_alfabetizacao_2024_uf` | Meta de alfabetização da UF para 2024 (rede pública) | INEP (Meta de Alfabetização Estadual) | — |

---

## Features binárias (2)

| Coluna | Descrição | Observação |
|---|---|---|
| `rede_Municipal` | 1 = rede Municipal, 0 = rede Estadual | One-hot de `rede` com `drop_first=True` (Estadual é a categoria de referência) |
| `meta_alfabetizacao_2024_municipio_disponivel` | 1 = valor real de `meta_alfabetizacao_2024_municipio`, 0 = valor imputado (mediana) | Indica quando a meta municipal não se aplica (100% da rede Estadual) ou não foi reportada (resíduo pequeno da rede Municipal) |

---

## Features categóricas codificadas — `sigla_uf` (22 colunas, one-hot com `drop_first=True`)

`AL` (Alagoas) é a categoria de referência (não gera coluna própria — está implícita quando todas as outras UFs abaixo são 0).

| Coluna | UF |
|---|---|
| `sigla_uf_AM` | Amazonas |
| `sigla_uf_AP` | Amapá |
| `sigla_uf_BA` | Bahia |
| `sigla_uf_CE` | Ceará |
| `sigla_uf_ES` | Espírito Santo |
| `sigla_uf_GO` | Goiás |
| `sigla_uf_MA` | Maranhão |
| `sigla_uf_MG` | Minas Gerais |
| `sigla_uf_MS` | Mato Grosso do Sul |
| `sigla_uf_MT` | Mato Grosso |
| `sigla_uf_PA` | Pará |
| `sigla_uf_PB` | Paraíba |
| `sigla_uf_PE` | Pernambuco |
| `sigla_uf_PI` | Piauí |
| `sigla_uf_PR` | Paraná |
| `sigla_uf_RJ` | Rio de Janeiro |
| `sigla_uf_RN` | Rio Grande do Norte |
| `sigla_uf_RO` | Rondônia |
| `sigla_uf_RS` | Rio Grande do Sul |
| `sigla_uf_SC` | Santa Catarina |
| `sigla_uf_SE` | Sergipe |
| `sigla_uf_TO` | Tocantins |

⚠️ **`SP`, `AC`, `RR` e `DF` não aparecem** — essas 4 UFs estão ausentes da base de origem (ver Limitações no README). O modelo não deve ser usado para esses estados.

---

## Colunas que existiam na Gold, mas foram removidas antes da modelagem

Documentado aqui por rastreabilidade — explica por que uma coluna que existe na tabela `alunos_alfabetizacao` (Gold da Fase 2/3) não aparece no `X` final.

### Removidas por data leakage

| Coluna | Motivo |
|---|---|
| `proficiencia` | Define o alvo diretamente (corte de 743 pontos) — correlação de 0,792, prova de match 100% |
| `taxa_alfabetizacao_municipio` | Agregado calculado incluindo o próprio aluno |
| `taxa_alfabetizacao_uf` | Mesmo motivo, em nível estadual |
| `media_portugues_municipio` | Mesmo mecanismo de `taxa_alfabetizacao_municipio` |
| `media_portugues_uf` | Mesmo mecanismo, em nível estadual |

### Removidas por multicolinearidade/redundância

| Coluna | Motivo |
|---|---|
| `pib` | Redundante com `pib_per_capita` (correlação 0,98 com população) |
| `quantidade_escolas_inse` | Artefato da agregação do INSE, não variável de negócio (correlação 0,99 com população) |
| `taxa_reprovacao_ef_2_ano_anterior` | Redundância matemática com `taxa_aprovacao` (correlação -0,99) |
| `icg_nivel_1` a `icg_nivel_6` | Correlação fraca com o alvo (< 0,03) e forte entre si (até 0,92); teste de "nível predominante" mostrou 97,1% de "quase empate" entre municípios |
| `meta_alfabetizacao_2024_brasil` | Variância zero (valor único de 59,9 para todo o Brasil em 2023) |

### Removidas por sinal fraco / nulo estrutural sem imputação confiável

| Coluna | Motivo |
|---|---|
| `had_ef_anos_iniciais` | 4,09% de nulo em padrão "tudo ou nada" por município (1.053 municípios sempre nulos), correlacionado com tamanho do município; correlação fraca com o alvo (0,05) |
| `peso_aluno` | Correlação próxima de zero com tudo (é peso de amostragem estatística, não característica do aluno); outlier extremo concentrado no Rio de Janeiro |

### Não usadas como feature (mas mantidas na Gold por transparência)

| Coluna | Motivo |
|---|---|
| `id_aluno`, `id_escola`, `id_municipio` | Identificadores, sem significado de quantidade — correlação espúria confirmada no heatmap |
| `ano`, `serie`, `presenca` | Variância zero após os filtros de escopo (só 2023, só 1 série, só "Presente") |
| `preenchimento_caderno` | Extremamente desbalanceada (249 casos de 1,5 milhão) |
| `sigla_uf_nome`, `id_municipio_nome` | Duplicatas textuais de `sigla_uf`/`id_municipio` |
| `_gold_processed_at` | Metadado técnico da pipeline, não variável de negócio |
