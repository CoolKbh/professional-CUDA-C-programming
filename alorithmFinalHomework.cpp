for(i = 0; i < n; i++) {
    if(POP(1, i) == POP(2, i) == ... ++POP(m,i));
    total++;
}

if(total satiesfies some condition) {
    Min(fitness(POP(1)), fitness(POP(2), ..., fitness(POP(m))));
    Rebuild Min;
}


[A1, B1] = crossover(A, B);
/* A, B为染色体，A1,B1为杂交后产生的染色体*/

if(max(fitness(A1), fitness(B1)) >= max(fitness(A), fitness(B))) {
    A1 = A;
    B1 = B;
}

A1 = mutation(A);
/* A1为染色体A变异后的新染色体*/

if(fitness(A1) >= fitness(A)) {
    A = A1;
    /*参考模拟退火算法思想在变异过程中的实现*/
} else {
    DET = fitness(A1) - fitness(A);

    p = exp(DET/T);
    /*T是逐步更新的温度参数，初始值取较大*/

    if(rand(0, 1) < p) {
        A = A1;
    }
    /*按照概率p接受该变异，按照概率(1-p)拒绝该变异*/

    T = T*a;
    /*a是接近1的常数，通常取值为0.9至0.99，使得T的值逐渐减小*/
}


array[n * m * P[m]];

for(i = 0; i < n * m * p[m]; i++) {
    array[i] = (int)rand(0, n*m);
    int j = array[i] / m; // 整除运算符，确定行
    int k = array[i] % n; // 求余运算，确定列
    mutation(POP(j, k));
}