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
    srand(42); // Semente fixa para resultados iguais sempre

    // 1. Gerar DADOS
    FILE *f_data = fopen("dados.csv", "w");
    for (int i = 0; i < N; i++) {
        // Cria clusters a cada 50.0 unidades
        double center = (rand() % K) * 50.0; 
        fprintf(f_data, "%.4f\n", center + rand_double(-10.0, 10.0));
    }
    fclose(f_data);

    // 2. Gerar CENTRÓIDES INICIAIS
    FILE *f_cent = fopen("centroides_iniciais.csv", "w");
    for (int i = 0; i < K; i++) {
        fprintf(f_cent, "%.4f\n", rand_double(0, K * 50.0));
    }
    fclose(f_cent);

    printf("Gerado: dados.csv (%d linhas) e centroides_iniciais.csv (%d linhas)\n", N, K);
    return 0;
}