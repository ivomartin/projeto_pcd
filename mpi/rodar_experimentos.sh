#!/bin/bash


RESULTADOS_CSV="resultados_gerais.csv"

mkdir -p saidas

echo "--- [1/3] Compilando os códigos... ---"

gcc gerar_dados.c -o gerar_dados
if [ $? -ne 0 ]; then echo "Erro ao compilar gerar_dados.c"; exit 1; fi

mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
if [ $? -ne 0 ]; then echo "Erro ao compilar kmeans_1d_mpi.c"; exit 1; fi

echo "Compilação OK!"

echo "Cenario,N,K,Processos,Iteracoes,SSE_Final,Tempo_s" > $RESULTADOS_CSV

rodar_cenario() {
    local N=$1
    local K=$2
    local PROC=$3
    local NOME=$4

    echo "------------------------------------------------------------"
    echo "Rodando Cenario: $NOME (N=$N, K=$K, P=$PROC)..."

    ./gerar_dados $N $K > /dev/null

    SAIDA=$(mpirun --allow-run-as-root -np $PROC ./kmeans_1d_mpi dados.csv centroides_iniciais.csv)

    echo "$SAIDA"

    LINHA=$(echo "$SAIDA" | grep "Fim:")
    ITER=$(echo "$LINHA" | awk