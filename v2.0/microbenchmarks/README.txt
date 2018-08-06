***********************************************************************************
*                                                                                 *
*                                MICROBENCHMARK SUITE                             *
*                                                                                 *
***********************************************************************************

1 - README

    Here is provided a suite of CUDA microbenchmark applications, extended from
    the one presented in [1], which can be used to characterize several of the
    main GPU components, namely the INT/FP32/FP64/SF/CF Units, the L2-cache,
    the shared memory and the DRAM).

    Please see [1] for more details on the structure of the GPU kernels and how these
    microbenchmarks can be used to estimate a power consumption model of the GPU.

    Also, if you use the set of microbenchmarks in a publication, please cite:

    [1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. GPGPU Power Modelling
        for Multi-Domain Voltage-Frequency Scaling. 24th IEEE International Symposium on
        High-Performance Computing Architecture (HPCA), 2018.

    [2] André Lopes, Frederico Pratas, Leonel Sousa, Aleksandar Ilic. Exploring GPU
        performance, power and energy-efficiency bounds with Cache-aware Roofline
        Modeling. 2017 IEEE International Symposium on Performance Analysis of Systems
        and Software (ISPASS), 2017.

2 - CONTACT

    If you have problems, questions, ideas or suggestions, please contact us by
    e-mail at joao.guerreiro@inesc-id.pt.

3 - USAGE

    To compile the microbenchmark use the following commands:
        > chmod 755 compile.py
        > ./compile.py

    The compilation script is by default considering a device with compute capability
    of 5.2. If using the microbenchmarks on a different device, adjust the variable
    arch in the compile.py file accordingly (e.g., arch=35 for a compute capability
    of 3.5).

    All microbenchmarks can be called in the following way:

        > ./microbenchmark [compute_iterations] [kernel_calls]

    The [compute_iterations] and [kernel_calls] arguments are optional. The number
    of compute_iterations can be changed to adjust the number of arithmetic operations
    for each data access - Arithmetic Intensity (see [1] for more details). Similarly,
    the number of kernel calls can also be adjusted as to adjust the number of successive
    kernel calls (useful when kernel execution time is low).
