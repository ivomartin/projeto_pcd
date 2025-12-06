# K-means 1D com MPI

Implementação paralela do algoritmo K-means para dados unidimensionais utilizando MPI (Message Passing Interface).

## Pré-requisitos

- GCC (compilador C)
- OpenMPI ou MPICH
- Python 3 com pandas e matplotlib (para visualização)

### Instalação das dependências (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install gcc openmpi-bin libopenmpi-dev
```

## Compilação

### Compilar o gerador de dados
```bash
gcc gerar_dados.c -o gerar_dados
```

### Compilar o algoritmo K-means MPI
```bash
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
```

## Execução

### 1. Gerar dados de teste
```bash
./gerar_dados <N_pontos> <K_clusters>
```
Exemplo:
```bash
./gerar_dados 1000000 16
```
Isso gera `dados.csv` (N pontos) e `centroides_iniciais.csv` (K centróides).

### 2. Executar o K-means MPI
```bash
mpirun -np <num_processos> ./kmeans_1d_mpi dados.csv centroides_iniciais.csv
```
Exemplo com 4 processos:
```bash
mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv
```

### 3. Executar bateria de testes completa
```bash
chmod +x rodar_bateria_testes.sh
./rodar_bateria_testes.sh
```
Os resultados serão salvos em `resultados_completo.csv`.

## Saídas

- `assign.csv`: Atribuição de cada ponto ao seu cluster
- `centroids.csv`: Posição final dos centróides
- `resultados_completo.csv`: Métricas de todos os experimentos

## Experimentos

A bateria de testes executa três experimentos:

| Experimento | Variável | Valores | Fixos |
|-------------|----------|---------|-------|
| Escalabilidade | Processos | 1, 2, 3, 4, 8, 16, 32, 64 | N=1M, K=16 |
| Variação de N | N (pontos) | 100k a 5M | K=16, P=4 |
| Variação de K | K (clusters) | 4 a 220 | N=500k, P=4 |

## Visualização

Os gráficos são gerados pelo notebook `plots/main.ipynb`:
- `grafico_Escalabilidade.png`
- `grafico_Variacao_N.png`
- `grafico_Variacao_K.png`
