# Tech Challenge Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e social do país. Gestores públicos precisam antecipar riscos, identificar regiões vulneráveis e compreender quais fatores possuem maior impacto nos indicadores educacionais.

Este projeto dá continuidade à Fase 2 (construção da pipeline de engenharia de dados do Indicador Criança Alfabetizada), utilizando a camada Gold, desenvolvida como base para modelagem supervisionada.

## Objetivo analítico

Desenvolver um modelo supervisionado capaz de prever se um aluno será considerado **alfabetizado ou não alfabetizado**, a partir de variáveis educacionais, territoriais e socioeconômicas, e usar essa modelagem para responder perguntas de negócio relevantes para políticas públicas educacionais.

## Descrição da base utilizada

A base parte da camada Gold construída na Fase 2 (`alunos_alfabetizacao`), enriquecida com fontes adicionais para atender aos pilares de dado exigidos pelo desafio. **O dicionário de dados completo (37 features + alvo, incluindo o que foi removido e por quê) está em [`docs/dicionario_dados.md`](docs/dicionario_dados.md).**

| Fonte | Pilar do desafio | Nível de granularidade |
|---|---|---|
| Indicador Criança Alfabetizada (avaliação de alfabetização) | Indicador-alvo | Aluno |
| Metas de Alfabetização (municipal, estadual, nacional) | Metas | Município / UF / Brasil |
| População (IBGE) | Territorial / populacional | Município |
| PIB dos Municípios (IBGE) | Socioeconômico | Município |
| INSE, Indicador de Nível Socioeconômico (INEP) | Socioeconômico | Escola (agregado para município) |
| Indicadores Educacionais (INEP) | Educacional complementar | Município |

Após tratamento (ver seção de limitações), a base final de modelagem contém **1.501.374 alunos e 37 features**, com alvo binário `alfabetizado` (Sim/Não).

## Etapas de modelagem

A pipeline seguiu, nessa ordem:

1. **Definição de escopo/coorte**: exclusão de alunos ausentes na avaliação (`presenca == "Ausente"`), pois o rótulo `alfabetizado = Não` desses alunos é uma convenção do sistema (sem nota registrada), não uma medição real da alfabetização.
2. **Tratamento de data leakage**: identificação e remoção de 5 colunas que vazavam o resultado do próprio aluno (`proficiencia`, que define o alvo diretamente; e 4 agregados de município/UF calculados incluindo o próprio aluno).
3. **Tratamento de multicolinearidade**: identificação de 3 clusters de colunas redundantes via matriz de correlação (população/PIB/quantidade de escolas do INSE; taxa de aprovação/reprovação; os 6 níveis do indicador de complexidade de gestão escolar), com decisão documentada para cada cluster.
4. **Tratamento de valores faltantes**: estratégias diferenciadas por causa raiz, `fillna(0)` quando o nulo reflete ausência real de uma categoria; descarte de coluna quando o nulo reflete indisponibilidade estrutural (ex.: município abaixo do critério mínimo de reporte) combinada com correlação fraca; indicador de disponibilidade mais a mediana quando o nulo é estrutural mas afeta uma fração grande da base (10%+) e não pode ser descartado.
5. **Feature engineering**: `pib_per_capita` (normalização de PIB por população), indicador de disponibilidade de meta municipal.
6. **Feature encoding**: one-hot encoding (com `drop_first=True`, evitando a "dummy variable trap") para `rede` e `sigla_uf`; `LabelEncoder` para o alvo.
7. **Separação treino/teste**: `train_test_split` com `stratify=y` (preserva a proporção do alvo) e `random_state` fixo (reprodutibilidade).
8. **Padronização**: `StandardScaler` ajustado (`fit`) exclusivamente no conjunto de treino, aplicado (`transform`) em treino e teste, evitando vazamento de informação do teste para o treino.
9. **Seleção e validação de features**: comparação entre 4 métodos independentes de importância de features (correlação, Informação Mútua, RFE, Feature Importance), buscando robustez nas conclusões.
10. **Validação cruzada e otimização de hiperparâmetros**: `RandomizedSearchCV` com validação cruzada (`cv=3`), otimizando diretamente o recall da classe de risco ("Não"), não a acurácia geral.
11. **Integração do pré-processamento ao pipeline do modelo**: reconstrução do modelo final como um único `Pipeline` (`ColumnTransformer` + `RandomForestClassifier`), corrigindo um vazamento sutil identificado no processo manual (a padronização era calculada usando todo o treino antes da validação cruzada dividir os dados em dobras no `Pipeline`, ela é recalculada a cada dobra).

## Organização dos notebooks: 
notebooks/eda.ipynb contém a análise exploratória completa e o raciocínio por trás de cada decisão (investigações de nulos, comparação de modelos, SHAP, etc.). notebooks/pipeline_final.ipynb contém a versão final, limpa e reproduzível, usando as funções modularizadas em src/ (preprocessing, modeling, evaluation, visualization) — reproduz os mesmos resultados finais em poucas células, sem repetir o processo de exploração.

## Escolha do algoritmo

Foram testados 4 modelos:

| Modelo | Acurácia (treino) | Acurácia (teste) |
|---|---|---|
| Regressão Logística | 64,38% | 64,40% |
| Árvore de Decisão (`max_depth=5`) | 64,49% | 64,46% |
| Árvore de Decisão (sem limite de profundidade) | 64,85% | 64,56% |
| **Random Forest (100 árvores)** | 64,85% | 64,56% |

**Modelo escolhido: Random Forest**, com `class_weight="balanced"`. Justificativa:
- Empatou na melhor acurácia entre os 4 modelos testados.
- Demonstrou maior estabilidade na importância de features quando comparado com uma única árvore de decisão (o padrão de importância de UFs específicas se manteve mais consistente ao usar 100 árvores em vez de 1).
- Permite interpretabilidade nativa via `feature_importances_` e SHAP (`TreeExplainer`).
- Captura padrões não-lineares e interações entre variáveis que a Regressão Logística (modelo linear) não conseguiu capturar, evidenciado pelo salto de desempenho da Regressão Logística ao receber as features de meta (ganho real), contra o ganho quase nulo da Random Forest (que já capturava sinal equivalente por outros caminhos).

Todos os 4 modelos convergiram para uma faixa de acurácia entre 64% e 65%, sugerindo um teto real de sinal extraível dos dados disponíveis, nenhuma feature individual apresentou correlação forte com o alvo (a mais forte, isolada, foi 0,333).

**Validação e otimização**: a busca de hiperparâmetros (amostra de 200 mil linhas, `RandomizedSearchCV`, `cv=3`, otimizando recall da classe "Não") encontrou uma Random Forest mais simples (`n_estimators=50`, `max_depth=10`) com desempenho estatisticamente equivalente ao modelo original (100 árvores, sem limite de profundidade): recall "Não" de 66% (vs. 67% original), acurácia de 62% (idêntica). A estabilidade entre a configuração original e a otimizada via validação cruzada confirma que o resultado reflete um padrão real nos dados, não uma coincidência do split treino/teste inicial e o modelo mais simples é preferível por eficiência computacional, sem perda de qualidade.

## Métricas de avaliação

Acurácia isolada mostrou-se insuficiente para avaliar este modelo, dado o objetivo de identificação de risco educacional. Comparação com e sem balanceamento de classe:

| Métrica | Sem balanceamento | Com `class_weight="balanced"` |
|---|---|---|
| Recall "Não alfabetizado" | 44% | **67%** |
| Recall "Alfabetizado" | 79% | 59% |
| Acurácia geral | 65% | 62% |

**Decisão**: adotado o modelo com balanceamento de classe. Para uma aplicação de política pública preventiva, o custo de não identificar um aluno em risco real (falso negativo) é maior que o custo de sinalizar atenção a um aluno que já ia bem (falso positivo) por isso, recall da classe de risco foi priorizado sobre acurácia geral.

## Interpretação dos resultados

A interpretabilidade foi construída triangulando 4 métodos independentes: correlação (Filter), Informação Mútua (Filter, robusto a relações não-lineares), RFE (Wrapper) e Feature Importance/SHAP (Embedded):

| Variável | Consistência entre métodos |
|---|---|
| `meta_alfabetizacao_2024_municipio` | 4 de 4 métodos |
| `meta_alfabetizacao_2024_uf` | 4 de 4 métodos |
| `dsu_ef_anos_iniciais` (% docentes com curso superior) | 3 de 4 métodos |
| `inse_medio` (nível socioeconômico) | 3 de 4 métodos |
| `rede` (Municipal vs. Estadual) | 2 de 4 métodos |

A convergência entre métodos independentes reforça a confiança nas conclusões, em vez de depender de uma única técnica.

## Insights encontrados

**1. Metas de alfabetização municipal/estadual são os preditores mais fortes.** Um município com meta historicamente baixa (reflexo de desempenho recente fraco) tende a ter alunos com pior resultado individual, a meta funciona como um resumo eficiente do contexto do município.

**2. Achado contraintuitivo: mais professores qualificados correlaciona com pior resultado.** `dsu_ef_anos_iniciais` (% de docentes com curso superior) mostrou correlação negativa com o alvo. Investigação revelou correlação de -0,48 com o nível socioeconômico do município, sugerindo que essa qualificação docente é direcionada como política compensatória a municípios mais vulneráveis, sem ser suficiente, sozinha, para reverter a desvantagem estrutural.

**3. Ceará se destaca com um padrão visual único na análise SHAP** um grupo de alunos com impacto fortemente positivo na predição, bem separado das demais UFs. Consistente com o Ceará ser nacionalmente reconhecido pelo PAIC (Programa Alfabetização na Idade Certa).

**4. Rede de ensino (Municipal vs. Estadual) tem diferença real, porém modesta**: 61,3% de alfabetização na rede Estadual contra 58,1% na Municipal, confirmado por dois métodos independentes (Informação Mútua e RFE).

**5. Adicionar uma feature forte nem sempre melhora a acurácia do modelo.** As metas de alfabetização, apesar de serem os preditores individuais mais fortes encontrados, praticamente não alteraram a acurácia da Random Forest (64,55% → 64,56%) a análise de `feature_importances_` mostrou que elas substituíram, sem adicionar, sinal que outras variáveis (`dsu_ef_anos_iniciais`, `inse_medio`, UF) já capturavam por caminhos indiretos.

## Limitações do projeto

- **Cobertura geográfica incompleta**: a base de alfabetização não contém dados de São Paulo, Acre, Roraima e Distrito Federal. São Paulo é o estado mais populoso do Brasil, o modelo não deve ser generalizado para essas 4 unidades federativas.
- **Incompatibilidade de identificador de escola**: o `id_escola` da avaliação de alfabetização não corresponde ao código INEP oficial usado em outras fontes (Censo Escolar, INSE) descoberto comparando os 2 primeiros dígitos do código (que deveriam representar a UF). Isso impediu enriquecimento no nível de escola; o INSE foi agregado por município como alternativa, com perda de granularidade documentada (mediana de 32 escolas por agregação, mas até 1.213 em alguns municípios).
- **`id_aluno` não é um identificador persistente entre anos**  o mesmo número se repete em edições diferentes da avaliação, referenciando alunos diferentes. Por isso a modelagem é transversal (um único ano-base, 2023), não uma previsão temporal.
- **Escopo temporal limitado a 2023**: a avaliação de alfabetização e a maioria das fontes de enriquecimento foram filtradas para um único ano, por decisão de escopo (classificação transversal) e por custo de extração. O modelo não incorpora evolução temporal real além do defasamento de 1 ano usado para evitar leakage.
- **Correlações individuais fracas**: nenhuma feature isolada teve correlação forte com o alvo (a mais forte, isolada, foi 0,333) reflexo de que fatores individuais de aluno (motivação, contexto familiar específico) não estão disponíveis na base, que é majoritariamente composta por variáveis de contexto agregado (município/escola).
- **Valores faltantes estruturais tratados com compromisso**: para `meta_alfabetizacao_2024_municipio` (ausente para 100% da rede Estadual, por não se aplicar a esse tipo de rede), optou-se por preencher com a mediana e criar uma coluna indicadora, em vez de descartar 10% da base.

## Aplicação prática para políticas públicas

O modelo permite:
- **Priorização de municípios para intervenção**: municípios com meta de alfabetização baixa, INSE baixo e alta taxa de abandono/distorção idade-série do ano anterior indicam maior concentração de risco educacional.
- **Identificação de programas estaduais bem-sucedidos** (como sugerido pelo padrão do Ceará/PAIC) como referência para replicação em outros estados com contexto socioeconômico semelhante.
- **Alerta antecipado de metas futuras**: a meta municipal já reflete uma trajetória de desempenho, municípios com meta baixa tendem a manter desempenho fraco, permitindo intervenção preventiva antes do resultado se concretizar.
- **Ressalva de uso**: dado o recall de 67% na classe de risco (não 100%), o modelo deve ser usado como **ferramenta de priorização e triagem**, não como substituto de avaliação pedagógica individual, aproximadamente 1 em cada 3 alunos em risco real ainda não é identificado pelo modelo.

## Possíveis evoluções futuras

- Enriquecer com metas nacionais/estaduais/municipais em série histórica (não só o próximo ano), permitindo uma análise de evolução temporal real por município.
- Resolver a incompatibilidade de `id_escola`, possivelmente obtendo um identificador oficial junto à fonte de dados, para permitir enriquecimento de Censo Escolar/INSE em granularidade real de escola.
- Explorar clusterização (por exemplo, K-means) sobre indicadores municipais para identificação sistemática de regiões com padrões semelhantes, complementando a inspeção UF por UF feita neste projeto.
- Testar ajuste de limiar de decisão como alternativa/complemento ao `class_weight`, para calibrar de forma mais fina o trade-off entre recall e precisão conforme a tolerância a risco do gestor público.

> As 5 perguntas de negócio do desafio (fatores de maior impacto, municípios de risco, regiões com padrões semelhantes, previsão de metas futuras, variáveis mais influentes) são respondidas em detalhe na seção "Aplicação Estratégica" de `notebooks/eda.ipynb`, e resumidas no vídeo executivo.