# OpenMP K-means Implementation

## Compilação
Para compilar usando OpenMP:
```bash
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
```

## Execution
Para executar:
- Configurar o número de threads com a variável de ambiente $env:OMP_NUM_THREADS = "8"
- Executar o comando:
```bash
./kmeans_1d_omp dados.csv centroides_iniciais.csv
```