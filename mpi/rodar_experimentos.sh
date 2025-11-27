cat << 'EOF' > rodar_experimentos.sh
#!/bin/bash

# Define nome do arquivo de resultados
RESULTADOS_CSV="resultados_gerais.csv"

# Cria pasta de saídas se não existir
mkdir -p saidas

echo "--- [1/3] Compilando os códigos... ---"

# Compila gerador
gcc gerar_dados.c -o gerar_dados
if [ $? -ne 0 ]; then echo "Erro ao compilar gerar_dados.c"; exit 1; fi

# Compila MPI
mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
if [ $? -ne 0 ]; then echo "Erro ao compilar kmeans_1d_mpi.c"; exit 1; fi

echo "Compilação OK!"

# Cria cabeçalho do CSV
echo "Cenario,N,K,Processos,Iteracoes,SSE_Final,Tempo_s" > $RESULTADOS_CSV

rodar_cenario() {
    local N=$1
    local K=$2
    local PROC=$3
    local NOME=$4

    echo "------------------------------------------------------------"
    echo "Rodando Cenario: $NOME (N=$N, K=$K, P=$PROC)..."

    # Gera dados
    ./gerar_dados $N $K > /dev/null

    # Roda MPI
    SAIDA=$(mpirun --allow-run-as-root -np $PROC ./kmeans_1d_mpi dados.csv centroides_iniciais.csv)

    # Mostra na tela
    echo "$SAIDA"

    # Filtra resultados
    LINHA=$(echo "$SAIDA" | grep "Fim:")
    
    # AQUI ESTAVA O SEU ERRO (Faltava fechar parênteses e completar):
    ITER=$(echo "$LINHA" | awk '{print $2}')
    SSE=$(echo "$LINHA" | awk '{print $4}' | sed 's/SSE=//' | sed 's/,//')
    TEMPO=$(echo "$LINHA" | awk '{print $5}' | sed 's/Tempo=//' | sed 's/s//')

    # Salva no CSV
    echo "$NOME,$N,$K,$PROC,$ITER,$SSE,$TEMPO" >> $RESULTADOS_CSV

    # Move arquivos
    mv assign.csv "saidas/assign_${NOME}.csv"
    mv centroids.csv "saidas/centroids_${NOME}.csv"
}

# --- EXECUÇÃO DOS CENÁRIOS ---

# Cenário 1: Pequeno
rodar_cenario 10000 4 4 "Pequeno"

# Cenário 2: Médio
rodar_cenario 100000 8 4 "Medio"

# Cenário 3: Grande
rodar_cenario 1000000 16 4 "Grande"

echo "============================================================"
echo "CONCLUÍDO!"
echo "Tabela de resultados: $RESULTADOS_CSV"
EOF