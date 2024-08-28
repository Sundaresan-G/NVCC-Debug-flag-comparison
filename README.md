## NVCC Debug flag comparisons

In this, we compare the effect of -G flag on the generated ptx, cubin and sass file by nvcc.

Configuration: NVIDIA V100 GPU with CUDA toolkit 12.1

Program used: A simple two vector addition program which utilises CUDA's float4 data structure.

1. nvcc -G flag leads to unoptimised code for both ptx and sass. This requires use of more registers and also more operations. Also we notice the term "debug" in .target:
```ptx
.version 8.1
.target sm_70, debug
.address_size 64
```
2. If -G and -dopt on are both enabled, only lineinfo is generated. This does not affect operations of ptx and sass. This is documented in nvcc --help too.
3. Now suppose we generate the ptx file with -G and then later generate .sass file (using ptxas and further followed by cuobjdump), this sass file is still equivalent to the one generated completely with -G flag. This is verified by comparing files "vec_add_float4_withDebugFlag_ptxasAssembledWithoutDebugFlag.sass" and "vec_add_float4_withDebugFlag.sass". Thus this experiment also shows that it is important to generate optimal ptx.
4. Dryrun (nvcc -dryrun) output results are also provided to understand the flags used at every stage. 