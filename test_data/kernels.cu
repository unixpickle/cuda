extern "C" __global__
void my_fancy_kernel(int n, float a, double b, int c, unsigned int d, double * out1, float * out2) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n) {
    out1[tid] = (double)tid + (double)a + b + (double)c;
		out2[tid] = (float)c + (float)d - a;
	}
}
