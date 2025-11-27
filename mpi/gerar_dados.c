#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Uso: %s <N_pontos> <K_clusters>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    // Se N ou K forem 0 (erro de argumento), avisa
    if (N <= 0 || K <= 0) {
        return 0; 
    }

    srand(42); 

    // 1. data
    FILE *f_data = fopen("dados.csv", "w");
    if(!f_data) return 1;
    for (int i = 0; i < N; i++) {
        double center = (rand() % K) * 50.0; 
        fprintf(f_data, "%.4f\n", center + rand_double(-10.0, 10.0));
    }
    fclose(f_data);
    printf("Gerado: dados.csv (%d linhas) ", N);

    // 2. centroids
    FILE *f_cent = fopen("centroides_iniciais.csv", "w");
    if(!f_cent) return 1;
    for (int i = 0; i < K; i++) {
        fprintf(f_cent, "%.4f\n", rand_double(0, K * 50.0));
    }
    fclose(f_cent);
    printf("e centroides_iniciais.csv (%d linhas)\n", K);

    return 0;
}