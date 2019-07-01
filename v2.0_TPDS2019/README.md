# gpmTOOL v2.0 - GPU  Power Modelling Tool


## 1. Description

``gpmTOOL`` is a command line tool for modelling the power consumption of
a GPU device. The tool implements the iterative heuristic algorithm proposed in [1] and [2], initially presented in [HPCA'2018](https://youtu.be/ppsPx6zaC0U), to determine the unknown characteristics of GPU devices in order to estimate the GPU power consumption across an ample range of frequency and voltage configurations for the multiple GPU frequency domains.

By providing an input file with details of the execution of different applications on a specific GPU device, the tool is able to model the characteristics of the different architecture components. The tool can also estimate how the cores voltage scales with their frequency. From the resulting model it is possible to estimate the power consumption of any new application for different frequency configurations, by providing the utilization of the modelled GPU components.

To get more information of how to install or use it, you should keep reading this file.

If you use the ``gpmTOOL`` tool in a publication, please cite [1] and [2].

## 2. Differences to v1.0
    * New microbenchmark suite.
    * Different iterative algorithm to estimate the parameters and core and memory voltages.
    * Please refer to the paper [2] for further details on how the new model can be utilized, namely using Scaling-Factors for better accuracy, or for exporting an estimated model to a different GPU device.

## 3. Contact

If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt.

## 4. Usage

* Usage:
```
gpmTOOL -t training_file.csv [-o model_file.csv] [-v]
gpmTOOL -p model_file.csv predictions_file.csv [-v]
gpmTOOL -u
```

* Options:

    ``-t`` : file with the measurements data points used to estimate the model parameters (eg. from microbenchmarks).

    ``-o`` : if provided, the determined model parameters and voltage values are saved in an output file model_file.csv.
    **ONLY SUPPORTED WHEN NUMBER OF FREQUENCY DOMAINS IS 2.**

    ``-p`` : the model provided in the model_file.csv is used to compute the power predictions for the applications provided in the predictions_file.csv. **ONLY SUPPORTED WHEN NUMBER OF FREQUENCY DOMAINS IS 2.**

    ``-v`` : verbose mode.

    ``-u`` : shows 'Usage' section of this file

## 5. Input Files

As proposed in [1], in order to estimate the power consumption model of a GPU device, i.e., to estimate the power consumption model, it is required to have information on the execution of different applications (eg. microbenchmarks) on the considered GPU device. During the execution of these applications, it is necessary to obtain both the power consumption at each tested frequency configuration, as well as the utilization of the different GPU components to be modeled (please see [5. Power Information](#5-power-information) and
[6. GPU Components Utilization](#6-gpu-components-utilization) for more details on how to obtain these values).

Once these values are measured, they can be aggregated in a file ([4.1. Training File](#41-training-file)), which can then be provided to the ``gpmTOOL`` in order to estimate the GPU power consumption model.

FOR CASES WHEN THE NUMBER OF FREQUENCY DOMAINS IS EQUAL TO 2, if desired, once the model is estimated it is possible to save the determined values into an output file ([4.2. Model File](#42-model-file)), which will contain both the estimated model parameters as well as the determined voltage values associated with each Frequency configuration.

Finally, it is also possible to use a previously estimated model to predict the power consumption of new (unseen) applications (ALSO REQUIRES NUMBER OF FREQUENCY DOMAINS EQUAL TO 2). For this, the tool must be provided both the previously created model file, as well as a file with the GPU components utilization for each of the desired GPU applications ([4.3. Predictions File](#43-predictions-file)).

All files are in .csv format, i.e. in each line the different values are separated by commas.

### 5.1. Training File

The training_file.csv must contain, at the beginning of the file, information relative to the GPU device, which will determine the basic structure of the model to be estimated (number_freq_domains, list_default_freqs, list_components). Afterwards, the file will
contain one line for each measurement to be used in the model estimation.

The training_file.csv must have the following format:

    line 1: number_freq_domains                                             -> single value
    line 2: list_default_freqs                                              -> length(list_default_freqs) = number_freq_domains, values separated by commas
    line 3: list_components                                                 -> length(list_components) = number_freq_domains
    line 4: components_names                                                -> length(components_names) = sum(list_components)
    line 5: Power_measure,list_freqs,list_utils_components                  -> length(list_freqs) = number_freq_domains; length(list_utils_components) = sum(list_components)
    ...
    ... (one line for each frequency configuration of each application to be used in the model estimation)

**Note:** Power consumption values should be in W, frequency values should be in MHz and utilization values should be in the [0,1] interval.

##### Example:

Considering the NVIDIA GTX Titan X used in [1], an example of the training_file.csv:

    line 1: 2                                                                                   -> 2 frequency domains (C - core and M - mem)
    line 2: 975,3505                                                                            -> Fc_default = 975MHz, Fm_default = 3505MHz
    line 3: 11,1                                                                                -> 11 components in the core domain, 1 component in the memory domain
    line 4: FP32 ADD,FP32 MUL,FP32 FMA,INT,FP64 ADD,FP64 MUL,FP64 FMA,SFU,CF,L2,Shared,DRAM     -> names of the modelled components
    line 5: 142.70,1164,4005,0.035,0.035,0.047,0.199,0,0,0,0,0.536,0.001,0,0.004                -> P,Fc,Fm,Uc0,Uc1,...,Uc10,Um0
    line 6: 138.363,1126,4005,0.035,0.035,0.047,0.199,0,0,0,0,0.536,0.001,0,0.004               -> P,Fc,Fm,Uc0,Uc1,...,Uc10,Um0
    ...

**Note:** that in this particular case (GTX Titan X) we have:
* Uc0 = Util_fp32_add_unit                      
* Uc1 = Util_fp32_mul_unit
* Uc2 = Util_fp32_fma_unit
* Uc3 = Util_int_unit
* Uc4 = Util_fp64_add_unit
* Uc5 = Util_fp64_mul_unit
* Uc6 = Util_fp64_fma_unit
* Uc7 = Util_sf_unit
* Uc8 = Util_cf_unit
* Uc9 = Util_l2_cache
* Uc10 = Util_shared_mem
* Um0 = Util_DRAM

### 5.2. Model File
##### (REQUIRES: number_freq_domains=2)

The model_file.csv must also contain, at the beginning of the file, information relative to the GPU device, which will determine the basic structure of the model to be estimated (number_freq_domains, list_default_freqs, list_components). Then the file will have the
different frequencies for each domain used during the model estimation (one line for each frequency domain). Afterwards, the model will have a line with the values of the determined model parameters. Finally, the file presents the determined voltage levels associated with each frequency configuration (each line corresponding to a different memory frequency).

The model_file.csv must have the following format:

    line 1: number_freq_domains                                                      -> single value
    line 2: list_default_freqs                                                       -> length(list_default_freqs) = number_freq_domains, values separated by commas
    line 3: list_components                                                          -> length(list_components) = number_freq_domains
    line 4: components_names                                                         -> length(components_names) = sum(list_components)
    line 5: list_freqs_domain_0                                                      -> length(list_freqs_domain_0) = number_freqs_domain_0
    line 6: list_freqs_domain_1                                                      -> length(list_freqs_domain_1) = number_freqs_domain_1
    line 7: list_parameters                                                          -> length(list_parameters) = 2*number_freq_domains + sum(list_components)
    line 8: list_voltages_domain_0                                                   -> length(list_voltages_domain_0) = number_freqs_domain_0
    line 9: list_voltages_domain_1                                                   -> length(list_voltages_domain_1) = number_freqs_domain_1

**Note:** The voltage values are relative to the voltage at the default frequency (see [1] for more details).

##### Example:

Considering the NVIDIA GTX Titan X used in [1], an example of the model_file.csv:

    line  1: 2                                                                                 -> 2 frequency domains (C - core and M - mem)
    line  2: 975,3505                                                                          -> Fc_default = 975MHz, Fm_default = 3505MHz
    line  3: 11,1                                                                              -> 11 components in the core domain, 1 component in the memory domain
    line  4: FP32 ADD,FP32 MUL,FP32 FMA,INT,FP64 ADD,FP64 MUL,FP64 FMA,SFU,CF,L2,Shared,DRAM               -> names of the modelled components
    line  5: 595,633,671,709,747,785,823,861,899,937,975,1013,1050,1088,1126,1164              -> 16 different core frequencies
    line  6: 810,3300,3505,4005                                                                -> 4 different memory frequencies
    line  7:                                                                                   -> Estimated parameters (beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3,omega4,omega5,omegam)
    line  8: 0.90501225,0.9050168,1.0,1.0993801                                                -> 16 voltage values for each core frequency
    line  9: 0.5,1.0,1.0,1.0                                                                   -> 4 voltage values for each memory frequency

### 5.3. Predictions File
##### (REQUIRES: number_freq_domains=2)

The predictions_file.csv must also contain, at the beginning of the file, information relative to the GPU device, which will determine the basic structure of the model to be estimated (number_freq_domains, list_default_freqs, list_components). Then, each following line will have the GPU component utilizations for a different application (measured at the default frequency configuration).

The predictions_file.csv must have the following format:

    line 1: number_freq_domains                                                      -> single value
    line 2: list_default_freqs                                                       -> length(list_default_freqs) = number_freq_domains, values separated by commas
    line 3: list_components                                                          -> length(list_components) = number_freq_domains
    line 4: components_names                                                         -> length(components_names) = sum(list_components)
    line 5: application_name_0,list_utils_application_0                              -> length(list_utils_application_0) = sum(list_components)
    line 6: application_name_1,list_utils_application_1                              -> length(list_utils_application_1) = sum(list_components)
    ...
    ... (one line for each different application)


**Note:** The voltage values are relative to the voltage at the default frequency (see [1] for more details).

##### Example:

Considering the NVIDIA GTX Titan X used in [1], an example of the model_file.csv:

    line 1: 2                                                                               -> 2 frequency domains (C - core and M - mem)
    line 2: 975,3505                                                                        -> Fc_default = 975MHz, Fm_default = 3505MHz
    line 3: 11,1                                                                            -> 11 components in the core domain, 1 component in the memory domain
    line 4: FP32 ADD,FP32 MUL,FP32 FMA,INT,FP64 ADD,FP64 MUL,FP64 FMA,SFU,CF,L2,Shared,DRAM -> names of the modelled components
    line 5: mri-gridding,0.125,0.040,0.158,0.215,0,0,0,0.054,0.241,0.038,0.201,0.028        -> bench_name,Uc0,...,Uc10,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
    line 6: cutcp,0.073,0.130,0.266,0.030,0,0,0,0.104,0.324,0.008,0.158,0.0004              -> bench_name,Uc0,...,Uc10,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
    line 7: lbm,0.027,0.011,0.036,0.030,0,0,0,0.006,0.012,0.175,0,0.582                     -> bench_name,Uc0,...,Uc10,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
    ...
    (Each line from line 5 forward corresponds to a different application, with utilizations are measured at the
    default frequency configuration)

Again in this particular case (GTX Titan X) we have:    
* Uc0 = Util_fp32_add_unit                      
* Uc1 = Util_fp32_mul_unit
* Uc2 = Util_fp32_fma_unit
* Uc3 = Util_int_unit
* Uc4 = Util_fp64_add_unit
* Uc5 = Util_fp64_mul_unit
* Uc6 = Util_fp64_fma_unit
* Uc7 = Util_sf_unit
* Uc8 = Util_cf_unit
* Uc9 = Util_l2_cache
* Uc10 = Util_shared_mem
* Um0 = Util_DRAM

## 6. Power Information

In NVIDIA GPU devices power samples can be obtained using the nvmlDeviceGetPowerUsage() function from the NVIDIA NVML library [3], which retrieves information from the Power sensor contained in some NVIDIA GPU devices.

We also provide a tool that uses this library to measure the power consumption of GPU applications [4].

## 7. GPU Components Utilization

In NVIDIA GPU devices, performance counters can be obtained during kernels execution using the CUPTI library [5]. Please see [1] for more details on what counter values should be measured and how to compute the corresponding component utilizations.

## 8. Example

In the example_files/ are provided examples of the input files (training, model and prediction), which correspond to the values presented in [1] for the NVIDIA GTX Titan X GPU. To estimate the power consumption model for this GPU and save it in a file called model_gtxtitanx.csv, use the following command:

``./gpmTOOL -t example_files/micro_gtxtitanx.csv -o model_gtxtitanx.csv``

To use the estimated model to predict the power consumptions of the real benchmarks applications considered in the file real_maxwell.csv, use the following command:

``./gpmTOOL -p model_gtxtitanx.csv example_files/real_gtxtitanx.csv``

## References

[1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. [GPGPU Power Modelling for Multi-Domain Voltage-Frequency Scaling](https://ieeexplore.ieee.org/abstract/document/8327055). 24th IEEE International Symposium on High-Performance Computing Architecture (HPCA), 2018.

[2] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. [Modeling and Decoupling the GPU Power Consumption for Cross-Domain DVFS](https://ieeexplore.ieee.org/abstract/document/8716300/).  IEEE Transactions on Parallel and Distributed Systems (TPDS), 2019.

[3] [NVIDIA. NVML API Reference Guide, vR384, 2017](http://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf)

[4] [gpowerSAMPLER, Power sampling tool](http://github.com/hpc-ulisboa/gpowerSAMPLER)

[5] [NVIDIA. NVIDIA CUPTI Users Guide, v9.1.85, 2017](http://docs.nvidia.com/cuda/pdf/CUPTI_Library.pdf)
