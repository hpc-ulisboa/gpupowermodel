# gpmTOOL - GPU Power Modelling Tool

## 1. DESCRIPTION
gpmTOOL is a command line tool for modelling the power consumption of a GPU device. The tool implements the iterative heuristic algorithm proposed in [1] and presented in [HPCA'2018](https://youtu.be/ppsPx6zaC0U), to determine the unknown characteristics of GPU devices in order to estimate the GPU power consumption across an ample range of frequency and voltage configurations for the multiple GPU frequency domains.

By providing an input file with details of the execution of different applications on a specific GPU device, the tool is able to model the characteristics of the different architecture components. The tool can also estimate how the cores voltage scales with their frequency. From the resulting model it is possible to estimate the power consumption of any new application for different frequency configurations, by providing the utilization of the modelled GPU components.

To get more information of how to install or use it, you should keep reading this file.

## 2. CONTACT

If you have problems, questions, ideas or suggestions, please contact us by e-mail at joao.guerreiro@inesc-id.pt .

## 3. USAGE

  * Usage:

    ``gpmTOOL -t training_file.csv [-o model_file.csv] [-v]``
    ``gpmTOOL -p model_file.csv predictions_file.csv [-v]``
    ``gpmTOOL -u``

  * Options:

    ``-t`` : file with the measurements data points used to estimate the model parameters (eg. from microbenchmarks).

    ``-o`` : if provided, the determined model parameters and voltage values are saved in an output file model_file.csv. ONLY SUPPORTED WHEN NUMBER OF FREQUENCY DOMAINS IS 2.

    ``-p`` : the model provided in the model_file.csv is used to compute the power predictions for the applications provided in the predictions_file.csv. ONLY SUPPORTED WHEN NUMBER OF FREQUENCY DOMAINS IS 2.

    ``-v`` : verbose mode.

    ``-u`` : shows 'USAGE' section of this file

## 4. INPUT FILES

As proposed in [1], in order to estimate the power consumption model of a GPU device, i.e., to estimate the power consumption model, it is required to have information on the execution of different applications (eg. microbenchmarks) on the considered GPU device. During the execution of these applications, it is necessary to obtain both the power consumption at each tested frequency configuration, as well as the utilization of the different GPU components to be modeled (please see [5. POWER INFORMATION](#5-power-information) and [6. GPU COMPONENTS UTILIZATON](#6-gpu-components-utilization) for more details on how to obtain these values).

Once these values are measured, they can be aggregated in a file ([4.1 - Training File](#41-training-file)), which can then be provided to the gpmTOOL in order to estimate the GPU power consumption model.

FOR CASES WHEN THE NUMBER OF FREQUENCY DOMAINS IS EQUAL TO 2, if desired, once the model is estimated it is possible to save the determined values into an output file ([4.2 - Model File](#42-model-file)), which will contain both the estimated model parameters as well as the determined voltage values associated with each Frequency configuration.

Finally, it is also possible to use a previously estimated model to predict the power consumption of new (unseen) applications (ALSO REQUIRES NUMBER OF FREQUENCY DOMAINS EQUAL TO 2). For this, the tool must be provided both the previously created model file, as well as a file with the GPU components utilization for each of the desired GPU applications ([4.3 - Predictions File](#43-predictions-file)).

All files are in .csv format, i.e. in each line the different values are separated by commas.

### 4.1. Training File

        The training_file.csv must contain, at the beginning of the file, information relative to
        the GPU device, which will determine the basic structure of the model to be estimated
        (number_freq_domains, list_default_freqs, list_components). Afterwards, the file will
        contain one line for each measurement to be used in the model estimation.

        The training_file.csv must have the following format:

            line 1: number_freq_domains                                             -> single value
            line 2: list_default_freqs                                              -> length(list_default_freqs) = number_freq_domains, values separated by commas
            line 3: list_components                                                 -> length(list_components) = number_freq_domains
            line 4: components_names                                                -> length(components_names) = sum(list_components)
            line 5: Power_measure,list_freqs,list_utils_components                  -> length(list_freqs) = number_freq_domains; length(list_utils_components) = sum(list_components)
            ...
            ... (one line for each frequency configuration of each application to be used in the model estimation)

            Note: Power consumption values should be in W, frequency values should be in MHz and utilization values should be in the [0,1] interval.

            Example:

                Considering the NVIDIA GTX Titan X used in [1], an example of the training_file.csv:

                line 1: 2                                                          -> 2 frequency domains (C - core and M - mem)
                line 2: 975,3505                                                   -> Fc_default = 975MHz, Fm_default = 3505MHz
                line 3: 6,1                                                        -> 6 components in the core domain, 1 component in the memory domain
                line 4: SP,INT,DP,SF,L2,Shared,DRAM                                -> names of the modelled components
                line 5: 225.2,1164,4005,0.65735,0.14174,0,0,0.11546,0,0.38509      -> P,Fc,Fm,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0
                line 6: 138.68,595,4005,0.65735,0.14174,0,0,0.11546,0,0.38509      -> P,Fc,Fm,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0
                ...

                Note that in this particular case (GTX Titan X) we have:    Uc0=Util_sp_unit
                                                                            Uc1=Util_int_unit
                                                                            Uc2=Util_dp_unit;
                                                                            Uc3=Util_sf_unit;
                                                                            Uc4=Util_l2_cache;
                                                                            Uc5=Util_shared_mem;
                                                                            Um0=Util_DRAM;

## 4.2. Model File (REQUIRES: number_freq_domains=2)

The model_file.csv must also contain, at the beginning of the file, information relative
to the GPU device, which will determine the basic structure of the model to be estimated
(number_freq_domains, list_default_freqs, list_components). Then the file will have the
different frequencies for each domain used during the model estimation (one line for each
frequency domain). Afterwards, the model will have a line with the values of the determined
model parameters. Finally, the file presents the determined voltage levels associated with
each frequency configuration (each line corresponding to a different memory frequency).

        The model_file.csv must have the following format:

            line 1: number_freq_domains                                                      -> single value
            line 2: list_default_freqs                                                       -> length(list_default_freqs) = number_freq_domains, values separated by commas
            line 3: list_components                                                          -> length(list_components) = number_freq_domains
            line 4: components_names                                                         -> length(components_names) = sum(list_components)
            line 5: list_freqs_domain_0                                                      -> length(list_freqs_domain_0) = number_freqs_domain_0
            line 6: list_freqs_domain_1                                                      -> length(list_freqs_domain_1) = number_freqs_domain_1
            line 7: list_parameters                                                          -> length(list_parameters) = 2*number_freq_domains + sum(list_components)
            line 8: list_voltages_fmem_0                                                     -> length(list_voltages_fmem_0) = number_freqs_domain_0
            ...
            line 8+(number_freqs_domain_1)-1: list_voltages_domain_(number_freqs_domain_1-1) -> length(list_voltages_domain_(number_freqs_domain_1-1)) = number_freqs_domain_0

        Note: The voltage values are relative to the voltage at the default frequency (see [1] for more details).

        Example:

            Considering the NVIDIA GTX Titan X used in [1], an example of the model_file.csv:

            line  1: 2                                                               -> 2 frequency domains (C - core and M - mem)
            line  2: 975,3505                                                        -> Fc_default = 975MHz, Fm_default = 3505MHz
            line  3: 6,1                                                             -> 6 components in the core domain, 1 component in the memory domain
            line  4: SP,INT,DP,SF,L2,Shared,DRAM                                      -> names of the modelled components
            line  5: 595,785,975,1164                                                -> 4 different core frequencies
            line  6: 810,3300,3505,4005                                              -> 4 different memory frequencies
            line  7: 24.44,0.0,0.015,0.012,0.056,0.048,0.028,0.126,0.069,0.041,0.015 -> Estimated parameters (beta0,beta1,beta2,beta3,omega0,omega1,omega2,omega3,omega4,omega5,omegam)
            line  8: 0.90501225,0.9050168,1.0,1.0993801                              -> 4 voltage values for each core frequency at Fm = 810MHz
            line  9: 1.0,1.0,1.0,1.1562239                                           -> 4 voltage values for each core frequency at Fm = 3300MHz
            line 10: 0.97783136,0.99070895,1.0,1.1435305                             -> 4 voltage values for each core frequency at Fm = 3505MHz
            line 11: 0.94551104,0.96700954,1.0,1.1404339                             -> 4 voltage values for each core frequency at Fm = 4005MHz

    4.3 - Predictions File (REQUIRES: number_freq_domains=2)

        The predictions_file.csv must also contain, at the beginning of the file, information
        relative to the GPU device, which will determine the basic structure of the model to
        be estimated (number_freq_domains, list_default_freqs, list_components). Then, each
        following line will have the GPU component utilizations for a different application
        (measured at the default frequency configuration).

        The predictions_file.csv must have the following format:

            line 1: number_freq_domains                                                      -> single value
            line 2: list_default_freqs                                                       -> length(list_default_freqs) = number_freq_domains, values separated by commas
            line 3: list_components                                                          -> length(list_components) = number_freq_domains
            line 4: components_names                                                         -> length(components_names) = sum(list_components)
            line 5: application_name_0,list_utils_application_0                              -> length(list_utils_application_0) = sum(list_components)
            line 6: application_name_1,list_utils_application_1                              -> length(list_utils_application_1) = sum(list_components)
            ...
            ... (one line for each different application)


        Note: The voltage values are relative to the voltage at the default frequency (see [1] for more details).

        Example:

            Considering the NVIDIA GTX Titan X used in [1], an example of the model_file.csv:

            line 1: 2                                                                   -> 2 frequency domains (C - core and M - mem)
            line 2: 975,3505                                                            -> Fc_default = 975MHz, Fm_default = 3505MHz
            line 3: 6,1                                                                 -> 6 components in the core domain, 1 component in the memory domain
            line 4: SP,INT,DP,SF,L2,Shared,DRAM                                         -> names of the modelled components
            line 5: streamcluster,0.0065652,0.046747,0,0,0.039836,0,0.10733             -> bench_name,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
            line 6: backprop,0.029646,0.1578,0.46532,0.13451,0.1052,0.03924,0.21002     -> bench_name,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
            line 7: lud,0.13084,0.18545,0,0.00017533,0.29286,0.76745,0.51052            -> bench_name,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
            line 8: 2mm,0.03251,0.034645,0,0,0.69131,0,0.13526                          -> bench_name,Uc0,Uc1,Uc2,Uc3,Uc4,Uc5,Um0 (measured at Fc_default = 975MHz, Fm_default = 3505MHz)
            ...
            (Each line from line 5 forward corresponds to a different application, with utilizations are measured at the
            default frequency configuration)

            Again in this particular case (GTX Titan X) we have:    Uc0=Util_sp_unit
                                                                    Uc1=Util_int_unit
                                                                    Uc2=Util_dp_unit;
                                                                    Uc3=Util_sf_unit;
                                                                    Uc4=Util_l2_cache;
                                                                    Uc5=Util_shared_mem;
                                                                    Um0=Util_DRAM;

5 - POWER INFORMATION

    In NVIDIA GPU devices power samples can be obtained using the nvmlDeviceGetPowerUsage()
    function from the NVIDIA NVML library [2], which retrieves information from the
    Power sensor contained in some NVIDIA GPU devices.

    We also provide a tool that uses this library to measure the power consumption of
    GPU applications [3].

6 - GPU COMPONENTS UTILIZATON

    In NVIDIA GPU devices, performance counters can be obtained during kernels execution
    using the CUPTI library [4]. Please see [1] for more details on what counter values
    should be measured and how to compute the corresponding component utilizations.

7 - Example

    In the example_files/ are provided examples of the input files (training, model
    and prediction), which correspond to the values presented in [1] for the NVIDIA GTX
    Titan X GPU. To estimate the power consumption model for this GPU and save it in a
    file called model_maxwell.csv, use the following command:

        > ./gpmTOOL -t example_files/micro_maxwell.csv -o model_maxwell.csv

    To use the estimated model to predict the power consumptions of the real benchmarks
    applications considered in the file real_maxwell.csv, use the following command:

        > ./gpmTOOL -p model_maxwell.csv example_files/real_maxwell.csv

---------------------------------------------------------------------------------------------------

[1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. GPGPU Power Modelling
    for Multi-Domain Voltage-Frequency Scaling. 24th IEEE International Symposium on
    High-Performance Computing Architecture (HPCA), 2018.

[2] NVIDIA. NVML API Reference Guide, vR384, 2017.
    http://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf

[3] João Guerreiro. gpowerSAMPLER, Power sampling tool.
    http://github.com/hpc-ulisboa/gpowerSAMPLER

[4] NVIDIA. NVIDIA CUPTI Users Guide, v9.1.85, 2017.
    http://docs.nvidia.com/cuda/pdf/CUPTI_Library.pdf
