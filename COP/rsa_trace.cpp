// rsa_trace.cpp — bit-exact scalar C++ tracer of the CUDA kernel's RSA dynamics.
// Builds a dense int8 J[N][N] from the same graph rule as ref_instance, runs the
// kernel's exact RSA pipeline for ONE replica, and prints "step energy" per line
// (energy AFTER each step; global_step starts at 1). Compile: g++ -O2.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static int N = 2048;
static const uint32_t P_MAX     = 0xFFFFFFFFu;
static const int      Z_CLAMPQ  = 2048;        // 8<<8

static const uint32_t SIG_P_ANCHOR[17] = {
    2147483648u,2673442470u,3139872686u,3511455636u,3782994643u,3969158893u,
    4091274721u,4169072223u,4217717111u,4247778736u,4266221719u,4277486187u,
    4284347459u,4288519766u,4291054360u,4292593129u,4293526977u};
static const uint32_t SIG_P_SLOPE[16] = {
    4109053u,3643986u,2902992u,2121398u,1454408u,954030u,607793u,380038u,
    234856u,144086u,88004u,53604u,32596u,19802u,12022u,7296u};

static inline uint32_t splitmix32(uint64_t x){
    x += 0x9E3779B97F4A7C15ull; uint64_t z=x;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return (uint32_t)(z>>32);
}

static inline uint16_t to_q8_8(double beta){
    double b = beta; if(b<0) b=0; if(b>255) b=255;
    double v = b*256.0;
    // round half away from zero
    double r = (v>=0)? std::floor(v+0.5) : std::ceil(v-0.5);
    return (uint16_t)((long long)r);
}

static inline uint32_t sigmoid_p_q32_from_zq(int zq){
    if(zq<=0) return 2147483648u;
    if(zq>=Z_CLAMPQ) return 4293526977u;
    int idx = zq/128; int rez = zq - idx*128;
    return (uint32_t)(SIG_P_ANCHOR[idx] + SIG_P_SLOPE[idx]*(uint32_t)rez);
}

static inline uint32_t p_plus_from_field_q32(int Li, uint16_t beta_q){
    if(Li==0) return 2147483648u;
    int absL = Li<0 ? -Li : Li;
    uint32_t zq = ((uint32_t)beta_q << 1) * (uint32_t)absL;
    if(zq > (uint32_t)Z_CLAMPQ) zq = (uint32_t)Z_CLAMPQ;
    uint32_t p = sigmoid_p_q32_from_zq((int)zq);
    if(Li<0) p = P_MAX - p;
    return p;
}

int main(int argc, char** argv){
    // args: stages iters_per_stage beta_start beta_end graph_seed spin_seed replica N
    int stages = 1, iters_per_stage = 2000;
    double beta_start = 4.0, beta_end = 4.0;
    uint64_t graph_seed = 1, spin_seed = 2;
    int replica = 0;
    if(argc>1) stages = atoi(argv[1]);
    if(argc>2) iters_per_stage = atoi(argv[2]);
    if(argc>3) beta_start = atof(argv[3]);
    if(argc>4) beta_end = atof(argv[4]);
    if(argc>5) graph_seed = strtoull(argv[5],0,10);
    if(argc>6) spin_seed = strtoull(argv[6],0,10);
    if(argc>7) replica = atoi(argv[7]);
    if(argc>8) N = atoi(argv[8]);

    // dense J
    std::vector<int8_t> J((size_t)N*N, -1);
    for(int i=0;i<N;i++) J[(size_t)i*N+i]=0;
    for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){
        uint64_t key = graph_seed ^ ((uint64_t)i<<21) ^ (uint64_t)j;
        if(splitmix32(key)&1u){ J[(size_t)i*N+j]=1; J[(size_t)j*N+i]=1; }
    }

    // beta schedule per stage
    std::vector<uint16_t> beta_q(stages);
    for(int k=0;k<stages;k++){
        double alpha = (stages>1)? (double)k/(double)(stages-1) : 1.0;
        double beta_f = (1.0-alpha)*beta_start + alpha*beta_end;
        beta_q[k] = to_q8_8(beta_f);
    }

    // init spins for this replica
    std::vector<int> s(N);
    for(int w=0;w<N/64;w++){
        uint64_t lo=splitmix32(spin_seed ^ ((uint64_t)replica<<32) ^ (uint64_t)w);
        uint64_t hi=splitmix32(spin_seed ^ ((uint64_t)replica<<40) ^ (uint64_t)w ^ 0xA5A5A5A5ull);
        uint64_t word=(hi<<32)|lo;
        for(int b=0;b<64;b++) s[w*64+b]= ((word>>b)&1ull)?1:-1;
    }
    // fields[i] = sum_{k!=i} J[i][k]*s[k]
    std::vector<int> fields(N,0);
    for(int i=0;i<N;i++){ long long Li=0; for(int k=0;k<N;k++) Li+=(long long)J[(size_t)i*N+k]*s[k]; fields[i]=(int)Li; }
    // energy = -0.5 * sum s[i]*fields[i]
    long long acc=0; for(int i=0;i<N;i++) acc += (long long)s[i]*fields[i];
    long long energy = -acc/2;

    uint64_t spin_seed64 = spin_seed;
    uint64_t r = (uint64_t)replica;
    for(int stage=0; stage<stages; stage++){
        for(int t=0; t<iters_per_stage; t++){
            uint64_t mix = spin_seed64 ^ ((uint64_t)r<<52) ^ ((uint64_t)stage<<32) ^ (uint64_t)t;
            uint32_t r_idx = splitmix32(mix);
            int j = (int)(((uint64_t)r_idx * (uint64_t)N) >> 32);
            uint32_t p_plus = p_plus_from_field_q32(fields[j], beta_q[stage]);
            bool old1 = (s[j]==1);
            uint32_t r_coin = splitmix32(mix ^ 0xD1B54A32D192ED03ull ^ (uint64_t)j);
            bool new1 = (r_coin < p_plus);
            if(new1 != old1){
                int s_old = old1 ? 1 : -1;
                energy += (long long)2*s_old*fields[j];
                s[j] = -s[j];
                for(int i=0;i<N;i++) if(i!=j) fields[i] -= 2*(int)J[(size_t)i*N+j]*s_old;
            }
            printf("%d %lld\n", stage*iters_per_stage + t + 1, energy);
        }
    }
    return 0;
}
