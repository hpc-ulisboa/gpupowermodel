# gpmTOOL - GPU Power Modelling Tool

gpmTOOL is a command line tool for modelling the power consumption of a GPU device. The tool implements the iterative heuristic algorithm proposed in [1] (https://youtu.be/ppsPx6zaC0U), to determine the unknown characteristics of GPU devices in order to estimate the GPU power consumption across an ample range of frequency and voltage configurations for the multiple GPU frequency domains.

By providing an input file with details of the execution of different applications on a specific GPU device, the tool is able to model the characteristics of the different architecture components. The tool can also estimate how the cores voltage scales with their frequency. From the resulting model it is possible to estimate the power consumption of any new application for different frequency configurations, by providing the utilization of the modelled GPU components.

To get more information of how to install or use it, you should keep reading this file.

Also, if you use the gpmTOOL tool in a publication, please cite:

[1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. GPGPU Power Modelling for Multi-Domain Voltage-Frequency Scaling. 24th IEEE International Symposium on High-Performance Computing Architecture (HPCA), 2018.

<br/>

## 1. CONTACT

    If you have problems, questions, ideas or suggestions, please contact us by
    e-mail at joao.guerreiro@inesc-id.pt .

## 2. VERSIONS

* [Version 1.0, originally released in January 2018 alongside the work in [1]] (https://github.com/hpc-ulisboa/gpupowermodel/tree/master/v1.0_HPCA2018)
* [Version 2.0, released in August 2018] (https://github.com/hpc-ulisboa/gpupowermodel/tree/master/v2.0). Major changes:
    * New microbenchmark suite.
    * Different iterative algorithm to estimate the parameters and core and memory voltages.

## 3. REFERENCES

    [1] João Guerreiro, Aleksandar Ilic, Nuno Roma, Pedro Tomás. GPGPU Power Modelling for Multi-Domain Voltage-Frequency Scaling. 24th IEEE International Symposium on High-Performance Computing Architecture (HPCA), 2018.

<br/>

## Author
João Guerreiro/ [@joaofilipedg](https://github.com/joaofilipedg)
