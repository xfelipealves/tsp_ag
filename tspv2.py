import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def gera_matriz_distancias(filepath):
    df = pd.read_csv(filepath, header=None, dtype=float)
    distancias = df.to_numpy()
    np.fill_diagonal(distancias, np.inf)
    return distancias

def gera_populacao_inicial(tamanho_pop, num_cidades):
    return [list(np.random.permutation(num_cidades)) for _ in range(tamanho_pop)]

def calcula_fitness(cromossomo, distancias):
    return sum(distancias[cromossomo[i], cromossomo[(i + 1) % len(cromossomo)]] for i in range(len(cromossomo)))

def seleciona_pais_roleta(populacao, fitnesses):
    total_fitness = sum(fitnesses)
    probabilidades = [f / total_fitness for f in fitnesses]
    pais = random.choices(populacao, weights=probabilidades, k=2)
    return pais[0], pais[1]

def seleciona_pais_torneio(populacao, fitnesses, tamanho_torneio):
    pais = []
    for _ in range(2):
        competidores = random.sample(list(zip(populacao, fitnesses)), tamanho_torneio)
        vencedor = min(competidores, key=lambda x: x[1])
        pais.append(vencedor[0])
    return pais[0], pais[1]

def seleciona_pais(populacao, fitnesses, metodo_selecao, **kwargs):
    return metodo_selecao(populacao, fitnesses, **kwargs)

def crossover_ordenado(pai1, pai2):
    tamanho = len(pai1)
    filho1, filho2 = [None]*tamanho, [None]*tamanho

    ponto1, ponto2 = sorted(random.sample(range(tamanho), 2))

    meio_pai1 = pai1[ponto1:ponto2 + 1]
    filho1[ponto1:ponto2 + 1] = meio_pai1

    pos_filho = (ponto2 + 1) % tamanho
    for elemento in pai2:
        if elemento not in meio_pai1:
            filho1[pos_filho] = elemento
            pos_filho = (pos_filho + 1) % tamanho

    meio_pai2 = pai2[ponto1:ponto2 + 1]
    filho2[ponto1:ponto2 + 1] = meio_pai2

    pos_filho = (ponto2 + 1) % tamanho
    for elemento in pai1:
        if elemento not in meio_pai2:
            filho2[pos_filho] = elemento
            pos_filho = (pos_filho + 1) % tamanho

    return filho1, filho2

def crossover(pai1, pai2, distancias):
    if pai1 == pai2:
        return pai1.copy()
    filho1, filho2 = crossover_ordenado(pai1, pai2)
    fit_filho1 = calcula_fitness(filho1, distancias)
    fit_filho2 = calcula_fitness(filho2, distancias)
    return filho1 if fit_filho1 < fit_filho2 else filho2

def mutacao(cromossomo, probabilidade_mutacao):
    if random.random() < probabilidade_mutacao:
        tipo_mutacao = random.choice(['swap', 'inversion', 'insertion'])
        if tipo_mutacao == 'swap':
            idx1, idx2 = random.sample(range(len(cromossomo)), 2)
            cromossomo[idx1], cromossomo[idx2] = cromossomo[idx2], cromossomo[idx1]
        elif tipo_mutacao == 'inversion':
            idx1, idx2 = sorted(random.sample(range(len(cromossomo)), 2))
            cromossomo[idx1:idx2+1] = cromossomo[idx1:idx2+1][::-1]
        elif tipo_mutacao == 'insertion':
            idx1, idx2 = random.sample(range(len(cromossomo)), 2)
            elemento = cromossomo.pop(idx1)
            cromossomo.insert(idx2, elemento)
    return cromossomo

def cria_nova_geracao(populacao, fitnesses, probabilidade_mutacao, distancias, elitismo=False):
    nova_populacao = []
    tamanho_pop = len(populacao)
    if elitismo:
        melhor_idx = np.argmin(fitnesses)
        melhor_individuo = populacao[melhor_idx].copy()
        nova_populacao.append(melhor_individuo)
    while len(nova_populacao) < tamanho_pop:
        pai1, pai2 = seleciona_pais(populacao, fitnesses, seleciona_pais_torneio, tamanho_torneio=3)
        filho = crossover(pai1, pai2, distancias)
        filho = mutacao(filho, probabilidade_mutacao)
        nova_populacao.append(filho)
    return nova_populacao

def algoritmo_genetico(tamanho_pop, num_geracoes, probabilidade_mutacao, distancias, num_cidades, elitismo=False):
    populacao = gera_populacao_inicial(tamanho_pop, num_cidades)
    melhor_fitness_por_geracao = []
    melhor_cromossomo = None
    melhor_fitness = float('inf')

    for geracao in range(num_geracoes):
        fitnesses = [calcula_fitness(ind, distancias) for ind in populacao]
        min_fitness = min(fitnesses)
        melhor_fitness_por_geracao.append(min_fitness)

        if min_fitness < melhor_fitness:
            melhor_fitness = min_fitness
            melhor_cromossomo = populacao[fitnesses.index(min_fitness)].copy()

        populacao = cria_nova_geracao(populacao, fitnesses, probabilidade_mutacao, distancias, elitismo)

    melhor_cromossomo = [int(cidade) for cidade in melhor_cromossomo]

    plt.figure(figsize=(10, 5))
    plt.plot(melhor_fitness_por_geracao, label='Melhor Fitness por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Progresso do Fitness ao Longo das Gerações')
    plt.legend()
    plt.grid(True)
    plt.show()

    return melhor_cromossomo, melhor_fitness

if __name__ == "__main__":
    filepath = r'distancias_entre_100_cidades.csv'  
    distancias = gera_matriz_distancias(filepath)
    num_cidades = len(distancias)
    tamanho_pop = 100
    num_geracoes = 1000
    probabilidade_mutacao = 0.01
    elitismo = False 

    # Opcional: definir sementes para reprodução de resultados
    # np.random.seed(42)
    # random.seed(42)

    melhor_cromossomo, melhor_fitness = algoritmo_genetico(
        tamanho_pop, num_geracoes, probabilidade_mutacao, distancias, num_cidades, elitismo
    )
    print("Melhor percurso:", melhor_cromossomo)
    print("Custo do melhor percurso:", melhor_fitness)
