#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// --- Funções de Leitura ---
static double *read_csv(const char *path, int *n_out){
    FILE *f = fopen(path, "r");
    if(!f) return NULL;
    int rows=0; char line[1024];
    while(fgets(line, sizeof(line), f)) if(strlen(line)>1) rows++;
    rewind(f);
    
    double *A = (double*)malloc(rows * sizeof(double));
    int r=0;
    while(fgets(line, sizeof(line), f) && r < rows){
        if(strlen(line)>1) A[r++] = atof(line);
    }
    fclose(f);
    *n_out = r;
    return A;
}

static void write_csv(const char *path, const int *data, int N){
    FILE *f = fopen(path, "w");
    if(!f) return;
    for(int i=0; i<N; i++) fprintf(f, "%d\n", data[i]);
    fclose(f);
}

static void write_centroids(const char *path, const double *C, int K){
    FILE *f = fopen(path, "w");
    if(!f) return;
    for(int i=0; i<K; i++) fprintf(f, "%.6f\n", C[i]);
    fclose(f);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0, K = 0;
    double *X_full = NULL;
    double *C = NULL;
    int max_iter = 50;
    double eps = 1e-4;

    // --- MESTRE (Rank 0) LÊ TUDO ---
    if (rank == 0) {
        if(argc < 3) { 
            fprintf(stderr, "Uso: %s dados.csv centroides.csv\n", argv[0]); 
            MPI_Abort(MPI_COMM_WORLD, 1); 
        }
        X_full = read_csv(argv[1], &N);
        C = read_csv(argv[2], &K);
        
        if(!X_full || !C) {
            fprintf(stderr, "Erro ao ler arquivos CSV (Rank 0)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // printf("Mestre: Lido N=%d, K=%d\n", N, K);
    }

    // --- COMUNICAÇÃO DE DADOS INICIAIS ---
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Todos alocam espaço para centróides
    if(rank != 0) C = (double*)malloc(K * sizeof(double));
    
    // Mestre envia centróides para todos
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- DISTRIBUIÇÃO DOS PONTOS (SCATTER) ---
    int local_N = N / size;
    int remainder = N % size;
    int *sendcounts = NULL; 
    int *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = (i < remainder) ? local_N + 1 : local_N;
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Ajusta tamanho local do processo atual
    int my_count = (rank < remainder) ? local_N + 1 : local_N;
    double *local_X = (double*)malloc(my_count * sizeof(double));
    int *local_assign = (int*)malloc(my_count * sizeof(int));

    // Espalha os dados
    MPI_Scatterv(X_full, sendcounts, displs, MPI_DOUBLE, 
                 local_X, my_count, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // Buffers auxiliares
    double *local_sum = (double*)malloc(K * sizeof(double));
    int *local_cnt = (int*)malloc(K * sizeof(int));
    double *global_sum = (double*)malloc(K * sizeof(double));
    int *global_cnt = (int*)malloc(K * sizeof(int));

    // --- LOOP PRINCIPAL DO K-MEANS ---
    double prev_sse = 1e300, global_sse = 0.0;
    int it = 0;
    double start = MPI_Wtime();

    for(it = 0; it < max_iter; it++) {
        double local_sse = 0.0;
        // Zera acumuladores
        for(int j=0; j<K; j++) { local_sum[j]=0.0; local_cnt[j]=0; }

        // 1. Assignment Local
        for(int i=0; i<my_count; i++) {
            double val = local_X[i];
            int best_c = 0;
            double min_dist = (val - C[0])*(val - C[0]);
            
            for(int c=1; c<K; c++){
                double d = (val - C[c])*(val - C[c]);
                if(d < min_dist) { min_dist = d; best_c = c; }
            }
            local_assign[i] = best_c;
            local_sse += min_dist;
            local_sum[best_c] += val;
            local_cnt[best_c]++;
        }

        // 2. Redução Global (MPI_Allreduce)
        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 3. Atualiza Centróides (Todos calculam igual)
        for(int c=0; c<K; c++) {
            if(global_cnt[c] > 0) C[c] = global_sum[c] / global_cnt[c];
        }

        // Convergência
        if(fabs(global_sse - prev_sse) < eps) { it++; break; }
        prev_sse = global_sse;
    }

    // --- RECOLHER RESULTADOS (GATHER) ---
    int *final_assign = NULL;
    if(rank == 0) final_assign = (int*)malloc(N * sizeof(int));

    MPI_Gatherv(local_assign, my_count, MPI_INT,
                final_assign, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // --- OUTPUT ---
    if (rank == 0) {
        printf("Fim: %d iterações, SSE=%.4f, Tempo=%.4fs\n", it, global_sse, MPI_Wtime()-start);
        write_csv("assign.csv", final_assign, N);
        write_centroids("centroids.csv", C, K);
        
        free(X_full); free(final_assign); free(sendcounts); free(displs);
    }
    
    free(C); free(local_X); free(local_assign);
    free(local_sum); free(local_cnt); free(global_sum); free(global_cnt);

    MPI_Finalize();
    return 0;
}
