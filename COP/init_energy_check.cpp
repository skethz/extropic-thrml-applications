#include <cstdint>
#include <cstdio>
#include <vector>
static constexpr int N = 2048;
static inline uint32_t splitmix32(uint64_t x){
    x += 0x9E3779B97F4A7C15ull; uint64_t z=x;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return (uint32_t)(z>>32);
}
int main(){
    static int8_t J[N][N];
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) J[i][j]= (i==j)?0:-1;
    for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){
        uint64_t key = 1ull ^ ((uint64_t)i<<21) ^ (uint64_t)j;
        if(splitmix32(key)&1u){ J[i][j]=1; J[j][i]=1; }
    }
    for(int r=0;r<4;r++){
        std::vector<int> s(N);
        for(int w=0;w<N/64;w++){
            uint64_t lo=splitmix32(2ull ^ ((uint64_t)r<<32) ^ (uint64_t)w);
            uint64_t hi=splitmix32(2ull ^ ((uint64_t)r<<40) ^ (uint64_t)w ^ 0xA5A5A5A5ull);
            uint64_t word=(hi<<32)|lo;
            for(int b=0;b<64;b++) s[w*64+b]= ((word>>b)&1ull)?1:-1;
        }
        int64_t acc=0; for(int i=0;i<N;i++){ int64_t Li=0; for(int j=0;j<N;j++) Li+=(int64_t)J[i][j]*s[j]; acc+=(int64_t)s[i]*Li; }
        printf("replica %d: E0 = %.1f\n", r, -0.5*(double)acc);
    }
}
