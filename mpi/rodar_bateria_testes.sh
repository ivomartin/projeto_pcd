cat << 'EOF' > rodar_bateria_testes.sh
#!/bin/bash

# ==============================================================================
# BATERIA DE TESTES AVANÇADA - PROJETO PCD
# ==============================================================================

RESULTADOS_CSV="resultados_completo.csv"

# 1. Limpeza e Compilação
echo "--- Preparando ambiente ---"
rm -rf saidas_bateria
mkdir -p saidas_bateria
rm -f dados.csv centroides_iniciais.csv $RESULTADOS_CSV

gcc gerar_dados.c -o gerar_dados
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

# Cabeçalho do CSV
echo "Experimento,N,K,Processos,Iteracoes,SSE_Final,Tempo_s" > $RESULTADOS_CSV

# Função auxiliar
executar() {
    local TIPO=$1
    local N=$2
    local K=$3
    local P=$4

    echo "[$TIPO] N=$N | K=$K | P=$P"
    
    # Gera dados (sobrescreve os anteriores para economizar disco)
    ./gerar_dados $N $K > /dev/null

    # Roda MPI (limitando a 1 thread por processo para medição mais real)
    SAIDA=$(mpirun --allow-run-as-root -np $P ./kmeans_1d_mpi dados.csv centroides_iniciais.csv)

    # Extrai dados
    LINHA=$(echo "$SAIDA" | grep "Fim:")
    ITER=$(echo "$LINHA" | awk '{print $2}')
    SSE=$(echo "$LINHA" | awk '{print $4}' | sed 's/SSE=//' | sed 's/,//')
    TEMPO=$(echo "$LINHA" | awk '{print $5}' | sed 's/Tempo=//' | sed 's/s//')

    echo "$TIPO,$N,$K,$P,$ITER,$SSE,$TEMPO" >> $RESULTADOS_CSV
}

# ==============================================================================
# EXPERIMENTO A: ESCALABILIDADE (SPEEDUP)
# Fixamos um problema grande e aumentamos os processadores.
# Isso mostra o ganho de usar MPI.
# ==============================================================================
echo "--- Iniciando Experimento A: Escalabilidade ---"
N_FIXO=1000000 # 1 Milhão
K_FIXO=16

# Testa com 1, 2, 3 e 4 processos (Se tiver mais cores, adicione 8, 16...)
for P in 1 2 3 4; do
    executar "Escalabilidade" $N_FIXO $K_FIXO $P
done

# ==============================================================================
# EXPERIMENTO B: CRESCIMENTO DE N (DADOS)
# Fixamos Processos e Clusters, aumentamos os dados.
# Isso deve gerar uma reta (linear).
# ==============================================================================
echo "--- Iniciando Experimento B: Variação de N ---"
K_FIXO=16
P_FIXO=4 

# De 100 mil até 2 Milhões
for N in 100000 500000 1000000 1500000 2000000; do
    executar "Variacao_N" $N $K_FIXO $P_FIXO
done

# ==============================================================================
# EXPERIMENTO C: COMPLEXIDADE DE K (CLUSTERS)
# Fixamos N e Processos, aumentamos a dificuldade (K).
# ==============================================================================
echo "--- Iniciando Experimento C: Variação de K ---"
N_FIXO=500000
P_FIXO=4

# 4, 16, 32, 64, 128 clusters
for K in 4 16 32 64 128; do
    executar "Variacao_K" $N_FIXO $K $P_FIXO
done

echo "============================================================"
echo "Bateria concluída!"
echo "Resultados salvos em: $RESULTADOS_CSV"
EOF