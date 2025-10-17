for openmp execution:
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
PS C:\projects\projeto_pcd\openmp> ./kmeans_1d_omp ..\data\dados.csv ..\data\centroides_iniciais.csv 8 