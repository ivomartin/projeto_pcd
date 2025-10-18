/* kmeans_1d_omp.c
   K-means 1D (C99 + OpenMP), implementação "naive" paralelizada.
   - Baseado na versão sequencial fornecida no projeto.
   - Paraleliza os passos de 'assignment' e 'update' com OpenMP.

   Compilar: gcc -O2 -std=c99 -fopenmp kmeans_1d_omp.c -o kmeans_1d_omp -lm
   Uso:      OMP_NUM_THREADS=4 ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Header para OpenMP

/* ---------- util CSV 1D: cada linha tem 1 número (sem alterações) ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- k-means 1D (versão OpenMP) ---------- */

/**
 * @brief Passo de atribuição (assignment) paralelizado com OpenMP.
 *
 * @param X Pontos de dados.
 * @param C Centróides atuais.
 * @param assign Array para armazenar o índice do cluster de cada ponto.
 * @param N Número de pontos.
 * @param K Número de clusters.
 * @return double Soma dos Erros Quadráticos (SSE).
 */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    // JUSTIFICATIVA DE PARALELIZAÇÃO:
    // O loop principal itera sobre todos os pontos de dados 'X'. A atribuição de
    // um ponto 'X[i]' a um cluster é uma operação totalmente independente das
    // outras. Cada thread pode, portanto, calcular o centróide mais próximo
    // para um subconjunto de pontos sem precisar de comunicação ou sincronização
    // com outras threads durante o cálculo.
    //
    // CLÁUSULA REDUCTION:
    // A variável 'sse' é um acumulador global. Para evitar uma condição de corrida
    // onde múltiplas threads tentam atualizar 'sse' simultaneamente, usamos a
    // cláusula `reduction(+:sse)`. O OpenMP cria uma cópia local de 'sse' para
    // cada thread, que acumula os erros localmente. Ao final da região paralela,
    // os valores de todas as cópias locais são somados (reduzidos) de forma
    // segura e atômica ao 'sse' global.
    #pragma omp parallel for reduction(+:sse)
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;
        // Este loop interno é pequeno (K << N) e sequencial para cada ponto.
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd; // A operação de redução ocorre aqui.
    }
    return sse;
}

/**
 * @brief Passo de atualização (update) paralelizado com OpenMP.
 *
 * @param X Pontos de dados.
 * @param C Centróides a serem atualizados.
 * @param assign Array com as atribuições de cluster para cada ponto.
 * @param N Número de pontos.
 * @param K Número de clusters.
 */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    // JUSTIFICATIVA DE PARALELIZAÇÃO:
    // Este loop, assim como no 'assignment', itera sobre todos os pontos 'X'.
    // A tarefa de cada iteração é acumular a soma e a contagem para o cluster
    // ao qual o ponto foi atribuído. Esta é outra operação "embaraçosamente paralela".
    //
    // CLÁUSULAS ATOMIC:
    // Diferente da etapa de 'assignment', aqui temos atualizações em posições
    // aleatórias dos arrays 'sum' e 'cnt'. Se duas threads processarem pontos
    // que pertencem ao mesmo cluster 'a', elas tentarão modificar `sum[a]` e
    // `cnt[a]` ao mesmo tempo, criando uma condição de corrida.
    // A diretiva `#pragma omp atomic` garante que a operação de atualização
    // (ex: `cnt[a] += 1`) seja executada de forma atômica, ou seja, sem que
    // outra thread possa interrompê-la. Isso é mais eficiente do que usar
    // uma seção 'critical' para todo o bloco, pois permite que threads que
    // atualizam clusters diferentes trabalhem em paralelo.
    #pragma omp parallel for
    for(int i=0;i<N;i++){
        int a = assign[i];
        #pragma omp atomic
        cnt[a] += 1;
        #pragma omp atomic
        sum[a] += X[i];
    }

    // Este segundo loop calcula a nova média para cada centróide.
    // Como K (número de clusters) é geralmente muito pequeno em comparação com N,
    // paralelizar este loop com OpenMP provavelmente introduziria mais sobrecarga
    // (overhead) do que ganho de desempenho. Por isso, ele é mantido sequencial.
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; // simples: cluster vazio recebe o primeiro ponto
    }
    free(sum); free(cnt);
}

/* ---------- Função principal do K-means (sem alterações) ---------- */
static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        printf("  Iter %d, SSE = %.6f, rel_var = %g\n", it+1, sse, rel);
        if(rel < eps){ it++; break; }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main (sem alterações, exceto pelo print) ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    // Usamos omp_get_wtime() para medição de tempo em paralelo
    double t0 = omp_get_wtime();
    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    double t1 = omp_get_wtime();
    double ms = 1000.0 * (t1 - t0);

    // Identifica a versão OpenMP no output
    printf("\nK-means 1D (OpenMP)\n");
    printf("N=%d K=%d max_iter=%d eps=%g threads=%d\n", N, K, max_iter, eps, omp_get_max_threads());
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}