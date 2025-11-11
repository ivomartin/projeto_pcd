/*
 * kmeans_1d_cuda.cu
 * K-means 1D (C99/CUDA), implementação com "assignment" na GPU.
 *
 * Compilar: nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
 * Uso: ./kmeans_1d_cuda dados.csv centroides_iniciais.csv [max_iter] [eps] [out_assign] [out_centroids]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

 // --- Macro para checagem de erros CUDA (Boa Prática) ---
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "Erro CUDA em %s, linha %d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                          \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// --- Funções de I/O de CSV (Sem Alterações) ---

static int count_rows(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        exit(1);
    }
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char* p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (!only_ws)
            rows++;
    }
    fclose(f);
    return rows;
}

static double* read_csv_1col(const char* path, int* n_out) {
    int R = count_rows(path);
    if (R <= 0) {
        fprintf(stderr, "Arquivo vazio: %s\n", path);
        exit(1);
    }
    double* A = (double*)malloc((size_t)R * sizeof(double));
    if (!A) {
        fprintf(stderr, "Sem memoria para %d linhas\n", R);
        exit(1);
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s\n", path);
        free(A);
        exit(1);
    }
    char line[8192];
    int r = 0;
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char* p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
                only_ws = 0;
                break;
            }
        }
        if (only_ws)
            continue;
        const char* delim = ",; \t";
        char* tok = strtok(line, delim);
        if (!tok) {
            fprintf(stderr, "Linha %d sem valor em %s\n", r + 1, path);
            free(A);
            exit(1);
        }
        A[r] = atof(tok);
        r++;
        if (r > R)
            break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char* path, const int* assign, int N) {
    if (!path)
        return;
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int i = 0; i < N; i++)
        fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char* path, const double* C, int K) {
    if (!path)
        return;
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s para escrita\n", path);
        return;
    }
    for (int c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

// --- Kernel CUDA para o Assignment Step ---

/*
 * Kernel de Assignment: 1 thread por ponto [cite: 50]
 * Cada thread (i) calcula o centróide (c) mais próximo para o seu ponto (X[i]).
 */
__global__ void assignment_kernel(const double* X, const double* C, int* assign,
    int N, int K) {
    // 1. Encontra o índice global da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Garante que a thread está dentro dos limites dos dados (N)
    if (i < N) {
        double bestd = 1e300;
        int best = -1;

        // 3. Varre os K centróides e encontra o mais próximo [cite: 51]
        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double d = diff * diff;
            if (d < bestd) {
                bestd = d;
                best = c;
            }
        }
        // 4. Escreve o resultado no vetor de atribuição
        assign[i] = best;
    }
}

// --- Funções K-means (CPU/Host) ---

/*
 * update_step_1d: (Executado no Host, sem alterações)
 * Recalcula a média dos pontos de cada cluster.
 */
static void update_step_1d(const double* X, double* C, const int* assign, int N,
    int K) {
    double* sum = (double*)calloc((size_t)K, sizeof(double));
    int* cnt = (int*)calloc((size_t)K, sizeof(int));
    if (!sum || !cnt) {
        fprintf(stderr, "Sem memoria no update\n");
        exit(1);
    }

    // Acumula somas e contagens
    for (int i = 0; i < N; i++) {
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }

    // Calcula novas médias
    for (int c = 0; c < K; c++) {
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
    free(sum);
    free(cnt);
}

/*
 * calculate_sse_host: (Nova função Host)
 * Calcula o SSE no host após copiar 'assign' da GPU. [cite: 53]
 */
static double calculate_sse_host(const double* X, const double* C,
    const int* assign, int N, int K) {
    double sse = 0.0;
    for (int i = 0; i < N; i++) {
        int a = assign[i];
        double diff = X[i] - C[a];
        sse += diff * diff;
    }
    return sse;
}

/*
 * kmeans_1d: (Função orquestradora - MODIFICADA)
 * Gerencia a memória e o loop de iteração.
 */
static void kmeans_1d(const double* X, double* C, int* assign, int N, int K,
    int max_iter, double eps, int* iters_out,
    double* sse_out, int blockSize) {

    // --- Alocação de Memória no Device (GPU) ---
    double* d_X, * d_C;
    int* d_assign;
    CUDA_CHECK(cudaMalloc((void**)&d_X, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (size_t)K * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_assign, (size_t)N * sizeof(int)));

    // --- Cópia Inicial Host -> Device (H2D) ---
    // Copia os pontos X (que são constantes) para a GPU apenas uma vez
    CUDA_CHECK(cudaMemcpy(d_X, X, (size_t)N * sizeof(double), cudaMemcpyHostToDevice));

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    // --- Loop de Iteração ---
    for (it = 0; it < max_iter; it++) {
        // 1. Copia centróides (C) atualizados do Host para o Device (H2D)
        //    (Para o kernel usar os valores corretos)
        CUDA_CHECK(cudaMemcpy(d_C, C, (size_t)K * sizeof(double), cudaMemcpyHostToDevice));

        // 2. Define configuração do Kernel
       // int blockSize = 16384; // Tamanho do bloco (experimente 128, 512, etc.) [cite: 57]
        int numBlocks = (N + blockSize - 1) / blockSize;

        // 3. Lança o Kernel de Assignment na GPU
        assignment_kernel << <numBlocks, blockSize >> > (d_X, d_C, d_assign, N, K);

        // Espera o kernel terminar
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Copia 'assign' do Device para o Host (D2H) [cite: 54]
        //    (Para o update_step e cálculo do SSE no host)
        CUDA_CHECK(cudaMemcpy(assign, d_assign, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

        // 5. Calcula SSE (no Host) [cite: 53]
        sse = calculate_sse_host(X, C, assign, N, K);

        // 6. Verifica convergência (no Host)
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if (rel < eps) {
            it++;
            break;
        }

        // 7. Executa Update (no Host) [cite: 54]
        update_step_1d(X, C, assign, N, K);

        prev_sse = sse;
    }

    *iters_out = it;
    *sse_out = sse;

    // --- Libera Memória do Device ---
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
}

// --- Função Main (Sem Alterações) ---

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [out_assign.csv] [out_centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabecalho\n");
        return 1;
    }
    int blockSize = atoi(argv[1]);
    const char* pathX = argv[2];
    const char* pathC = argv[3];
    int max_iter = (argc > 4) ? atoi(argv[4]) : 5000;
    double eps = (argc > 5) ? atof(argv[5]) : 1e-4;
    const char* outAssign = (argc > 6) ? argv[6] : NULL;
    const char* outCentroid = (argc > 7) ? argv[7] : NULL;

    if (max_iter <= 0 || eps <= 0.0) {
        fprintf(stderr, "Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N = 0, K = 0;
    double* X = read_csv_1col(pathX, &N);
    double* C = read_csv_1col(pathC, &K);
    int* assign = (int*)malloc((size_t)N * sizeof(int));
    if (!assign) {
        fprintf(stderr, "Sem memoria para assign\n");
        free(X);
        free(C);
        return 1;
    }

    clock_t t0 = clock();
    int iters = 0;
    double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse, blockSize);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA - Opcao A)\n");
    printf(" N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iteracoes: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign);
    free(X);
    free(C);
    return 0;
}

//PS C : \Users\ivoma\source\repos\CudaRuntime2 > C:\Users\ivoma\source\repos\CudaRuntime2\x64\Debug\CudaRuntime2.exe "1024" "C:\projects\projeto_pcd\openmp\dados.csv" "C:\projects\projeto_pcd\openmp\centroides_iniciais.csv"