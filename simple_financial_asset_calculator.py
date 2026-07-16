import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def toy_model_gerenciador_de_ativos(ativos, pesos, retornos_medios, volatilidades, corr,
    valor_inicial, reserva, percentual_seguranca, n_cenarios, seed=123):

    # Checagem de que as entradas são vetores.
    if len(ativos.shape) != 1:
        raise ValueError("A entrada em ativos deve ser um vetor.")
    if len(pesos.shape) != 1:
        raise ValueError("A entrada em pesos deve ser um vetor.")
    if len(retornos_medios.shape) != 1:
        raise ValueError("A entrada em retornos_medios deve ser um vetor.")
    if len(volatilidades.shape) != 1:
        raise ValueError("A entrada em volatilidades deve ser um vetor."

    # Checagens de que os valores são numéricos
    # Obs: a checagem de que as entradas da matriz de covariancia normalizada
    # é realizado indiretamente  no teste dos autovalores.
    assert all(isinstance(entrada, (int, float, np.integer, np.floating)) for entrada in pesos), "Os pesos devem ser numéricos."
    assert all(isinstance(entrada, (int, float, np.integer, np.floating)) for entrada in retornos_medios), "Os retornos médios devem ser numéricos."
    assert all(isinstance(entrada, (int, float, np.integer, np.floating)) for entrada in volatilidades), "As volatilidades devem ser numéricas."
    assert isinstance(valor_inicial, (int, float, np.integer, np.floating)), "O valor inicial deve ser numérico."
    assert isinstance(reserva, (int, float, np.integer, np.floating)), "A reserva deve ser numérica."

    # Checagem de que as proporções são de fato proporções.
    assert np.isclose(pesos.sum(), 1.0), "Os pesos devem somar 1."

    # Checagem se a porcentagem de segurança é uma porcentagem.
    assert 0 <= percentual_seguranca <= 100, "A porcentagem de segurança deve estar entre 0 e 100."

    # Checagens de formato
    assert pesos.shape == retornos_medios.shape == volatilidades.shape, "Dimensões incompatíveis."
    assert corr.shape == (len(ativos), len(ativos)), "Matriz de correlação com formato inválido."

    # Checagens particulares da matriz de correlação
    assert np.allclose(corr,corr.T), "A matriz de correlação deve ser simétrica."
    assert np.all(np.linalg.eigvals(corr)>0), "A matriz de correlação deve ser definida positiva."
    assert all(np.isclose(entrada, 1.0) for entrada in np.diag(corr)), "Os elementos na diagonal da matriz de correlações normalizadas devem ser 1.0."

    # Introdução de um DataFrame básico com os inputs relevantes dos ativos
    parametros_ativos = pd.DataFrame({
        "ativo": ativos,
        "peso": pesos,
        "retorno_medio": retornos_medios,
        "volatilidade": volatilidades
    })

    # Iniciar a sequência "aleatória" com um seed dado permite uma auditoria
    # mais fácil da função.
    rng = np.random.default_rng(seed)

    # Volatilidade = desvio padrão, nesse caso, a matriz de covariancia é dada
    # pelas correlações (normalizadas), que são a entrada da função,
    # multiplicada pelas volatilidades corretas; donde o outer-product.
    cov = np.outer(volatilidades, volatilidades) * corr

    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(n_cenarios, len(ativos)))

    # Modelo de simulação de retornos introduzido em aula.
    retornos_simulados = retornos_medios + z @ L.T

    # Ainda que muito improvável matematicamente, me parece razoável impor que
    # seja impossível perder mais dinheiro do que o investido.
    retornos_simulados = np.clip(retornos_simulados, -0.999, None)

    # Cálculo do retorno dada a distribuição de ativos na carteira
    retorno_carteira = retornos_simulados @ pesos

    valor_final = valor_inicial * (1 + retorno_carteira)
    perda = valor_inicial - valor_final

    # Cálculo indireto da reserva optimal dada a margem de risco desejada.
    var_margem = np.percentile(perda, percentual_seguranca)

    quebra = perda > reserva

    # Checagens simples sobre os formatos operados dentro da função.
    assert retornos_simulados.shape == (n_cenarios, len(ativos))
    assert retorno_carteira.shape == (n_cenarios,)
    assert valor_final.shape == (n_cenarios,)
    assert perda.shape == (n_cenarios,)

    # DataFrame com os retornos simulados distribuídos pelos ativos.
    df_cenarios = pd.DataFrame(
        retornos_simulados,
        columns=[f"retorno_{a}" for a in ativos]
    )

    # Introdução de uma primeira coluna indexando os cenários.
    df_cenarios.insert(0, "cenario", np.arange(1, n_cenarios + 1))

    # Informações importantes adicionadas ao DataFrame de cenários.
    df_cenarios["retorno_carteira"] = retorno_carteira
    df_cenarios["valor_final"] = valor_final
    df_cenarios["perda"] = perda
    df_cenarios["quebra"] = quebra

    # DataFrame menor analizando só casos em que quebra==True.
    df_quebras = df_cenarios[df_cenarios["quebra"]].copy()
    df_quebras["excesso_sobre_reserva"] = df_quebras["perda"] - reserva
    df_quebras = df_quebras.sort_values("perda", ascending=False)

    return var_margem, df_cenarios, df_quebras
