#!/bin/bash

RESULTADOS_CSV="resultados_completo.csv"

# cleaning
echo "--- Preparando ambiente ---"
rm -rf saidas_bateria
mkdir -p saidas_bateria
rm -f dados.csv centroides_iniciais.csv $RESULTADOS_CSV

#compiling
gcc gerar_dados.c -o gerar_dados
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

# csv head
echo "Experimento,N,K,Processos,Iteracoes,SSE_Final,Tempo_s" > $RESULTADOS_CSV

# main function
executar() {
    local TIPO=$1
    local N=$2
    local K=$3
    local P=$4

    echo "[$TIPO] N=$N | K=$K | P=$P"
    
    # generate data
    ./gerar_dados $N $K > /dev/null

    # runs mpi and gets the output
    SAIDA=$(mpirun --allow-run-as-root --oversubscribe -np $P ./kmeans_1d_mpi dados.csv centroides_iniciais.csv)

    # extract data
    LINHA=$(echo "$SAIDA" | grep "Fim:")
    ITER=$(echo "$LINHA" | awk '{print $2}')
    SSE=$(echo "$LINHA" | awk '{print $4}' | sed 's/SSE=//' | sed 's/,//')
    TEMPO=$(echo "$LINHA" | awk '{print $5}' | sed 's/Tempo=//' | sed 's/s//')

    echo "$TIPO,$N,$K,$P,$ITER,$SSE,$TEMPO" >> $RESULTADOS_CSV
}

echo "--- Iniciando Experimento A: scalability ---"
N_FIXO=1000000 # 1 Milhão
K_FIXO=16

# testing with 1, 2, 3 e 4 process ( can add more )
for P in 1 2 3 4 8 16 32 64; do
    executar "Escalabilidade" $N_FIXO $K_FIXO $P
done

echo "--- Iniciando Experimento B: Variação de N ---"
K_FIXO=16
P_FIXO=4 

# De 100 mil até 2 Milhões
for N in 100000 500000 1000000 1500000 2000000 3000000 5000000; do
    executar "Variacao_N" $N $K_FIXO $P_FIXO
done

echo "--- Iniciando Experimento C: Variação de K ---"
N_FIXO=500000
P_FIXO=4

# 4, 16, 32, 64, 128 clusters
for K in 4 16 32 64 128 160 200 220; do
    executar "Variacao_K" $N_FIXO $K $P_FIXO
done

echo "============================================================"
echo "Bateria concluída!"
echo "Resultados salvos em: $RESULTADOS_CSV"
