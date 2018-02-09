#!/usr/bin/python2
# TESTED USING PYTHON 2.7
import os.path
import sys
import warnings
import time
import csv #tested with version 1.0
import numpy as np #tested with version 1.14.0
import argparse #tested with version 1.1
import matplotlib.pyplot as plt #tested with version 2.1.1
from scipy.optimize import least_squares #tested with version 1.0.0
from scipy.optimize import nnls #tested with version 1.0.0

#============================ Definition of variables ============================#

verbose = 0

max_iterations = 500
threshold = 0.05

ymax_V_plot = 1.2

max_P = 250
bar_width = 0.6
bar_print_threshold = 2 #only put the percentage in the bars of values above this percentage
bar_min_Y = 50

benchs_per_row = 5

#============================ Definition of used functions ============================#

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '|' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def readFirstThreeLines(reader):
    # Parse the first line with the number of frequency domains
    line_1 = (next(reader))
    if (len(line_1) != 1):
        print "Wrong format on line 1"
        sys.exit()
    num_freq_domains = int(line_1[0])
    print "Num frequency domains = {}".format(num_freq_domains)

    # Parse the second line with the default frequency of each domain
    line_2 = (next(reader))
    if len(line_2) != num_freq_domains:
        print "Wrong format on line 2 (Must have one value for each frequency domain)"
        print "len(line_2): {} != num_freq_domains: {} ".format(len(line_2), num_freq_domains)
        sys.exit()

    default_freqs = np.zeros(num_freq_domains, dtype=np.int32)
    for domain_id,default_freq in enumerate(line_2):
        default_freqs[domain_id] = int(default_freq)
        print "Default frequency of domain {}: {}".format(domain_id, default_freq)

    # Parse the thir line with the number of components of each frequency domain
    line_3 = (next(reader))
    if len(line_3) != num_freq_domains:
        print "Wrong format on line 3 (Must have one value for each frequency domain)"
        print "len(line_3): {} != num_freq_domains: {} ".format(len(line_3), num_freq_domains)
        sys.exit()

    num_components_domains = np.zeros(num_freq_domains, dtype=np.int32)
    for domain_id,domain_size in enumerate(line_3):
        num_components_domains[domain_id] = int(domain_size)
        print "Size of domain {}: {}".format(domain_id, domain_size)
    return (num_freq_domains, default_freqs,  num_components_domains)

def fun_V(V, x1, x2, x3, P):
    return ((V * x1 + V * V * x2 + x3) - P)

def findLines_OneConfig(config):
    return  np.where((F==config).all(axis=1))

def findLines_AllConfigs(configs_list):
    lines_aux=[]
    for config_id in range(0, len(configs_list)):
        aux = findLines_OneConfig(np.asarray(configs_list, dtype=np.int32)[config_id])
        lines_aux.extend(aux)
    lines_aux = [item for sublist in lines_aux for item in sublist]

    return  np.unique(np.asarray(lines_aux, dtype=np.int32))

def printBCoefficients(array, num_domains, num_components_domains, tabs):
    # for i,coeff in enumerate(array):
    for idx in range(0, 2*num_domains):
        coeff = array[idx]
        if(tabs == 1):
            s = '\t['
        else:
            s = '\t\t['

        s += '{:7.4f}]  <--- '.format(coeff)

        s += 'beta_{} (Pconstant)'.format(idx)
        print s

    idx_aux = 0
    for domain_id in range(0, num_domains):
        for component_id in range(0, num_components_domains[domain_id]):
            coeff = array[idx_aux+2*num_domains]
            if(tabs == 1):
                s = '\t['
            else:
                s = '\t\t['

            s += '{:7.4f}]  <--- '.format(coeff)

            s += 'omega_{} (Pdynamic_domain{}_component{})'.format(idx_aux, domain_id, component_id)

            print s
            idx_aux += 1

def printArray_ints(array):
    s = '\t['
    for i, f in enumerate(array):
        if (i>0):
            s+=', '
        s += '{:4d}'.format(f)
    s += ']'
    print s

def print2DArray_ints(array):
    for i in array:
        printArray_ints(i)

def printArray_floats(array, tabs):
    s = '['
    for i, f in enumerate(array):
        if (i>0):
            s+=','
        s += '{:7.4f}'.format(f)
    s += ']'
    print s

def print2DArray_floats(array, tabs):
    for i in array:
        printArray_floats(i, tabs)

def printVoltage(array, core_freqs, mem_freqs, tabs):
    s = "\t{:<3}Fcore [MHz]:".format('')
    for clock_id, clock in enumerate(core_freqs):
        if clock_id > 0:
            s+=', '
        s += "{:7d}".format(clock)
    print s

    for mem_freq_id, voltage_line in enumerate(array):
        if (tabs == 2):
            s = '\t\t'
        else:
            s = '\t'

        s += 'Fmem={:4d}MHz: ['.format(mem_freqs[mem_freq_id])
        for voltage_id,v in enumerate(voltage_line):
            if (voltage_id > 0):
                s += ', '
            s += '{:7.4f}'.format(v)
        s += ']'
        print s

def printPowerBreakdown(P_breakdown):
    for row_id, row in enumerate(P_breakdown):
        s = '\tB{:>3}: TotalP = {:5.1f}, Breakdown: ['.format(row_id+1, np.sum(P_breakdown[row_id,:]))
        for util_id, util in enumerate(P_breakdown[row_id,:]):
            if (util_id > 0):
                s += ', '
            s += '{:4.1f}'.format(P_breakdown[row_id, util_id] / np.sum(P_breakdown[row_id,:])*100)
        s += ']%'
        print s

def printUsage():
    print "\nUsage:"
    print "\tgpowerModelTOOL -t training_file.csv [-o model_file.csv] [-v]"
    print "\tgpowerModelTOOL -p model_file.csv predictions_file.csv [-v]"
    print "\tgpowerModelTOOL -h"
    print "\tgpowerModelTOOL -u"

    print "\nOptions:"
    print "\t-t : file with the measurements data points used to estimate the model parameters (eg. from microbenchmarks)."
    print "\t-o : if provided, the determined model parameters and voltage values are saved in an output file model_file.csv. (REQUIRES NUMBER OF FREQUENCY DOMAINS = 2)."
    print "\t-p : the model provided in the model_file.csv is used to compute the power predictions for the applications provided in the predictions_file.csv. (REQUIRES NUMBER OF FREQUENCY DOMAINS = 2)."
    print "\t-v : verbose mode."
    print "\t-h : shows how the tool can be used."
    print "\t-u : shows the detailed usage guide of the tool.\n\n"

#============================ SOME STYLE FORMATTING ============================#
plt.style.use('ggplot')
warnings.simplefilter(action='ignore', category=FutureWarning)

# Tableau 20 Colors
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Rescale to values between 0 and 1
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#============================ VERIFY INPUT ARGUMENTS ============================#
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()

group.add_argument('-t', dest='training_file', type=lambda x: is_valid_file(parser, x), help='input file with data points to estimate the model', metavar="TRAIN_FILE")
parser.add_argument('-o',  dest='output_file', help='output file where to save the estimate model and voltage values', metavar="OUTPUT_FILE")
group.add_argument('-p', nargs=2, dest='model_predict_files', type=lambda x: is_valid_file(parser, x), help='files with the estimated model and benchmarks to make predictions on', metavar=('MODEL_FILE', 'PREDICT_FILE'))
parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
group.add_argument('-u', dest='usage', action='store_true', help='displays detailed usage guide')

args = parser.parse_args()

if (args.training_file != None):
    program_mode = 1
    f = args.training_file
elif (args.model_predict_files != None):
    program_mode = 2
    f1 = args.model_predict_files[0]
    f2 = args.model_predict_files[1]
elif (args.usage != False):
    printUsage()
    sys.exit()
else:
    print "Wrong argument usage."
    printUsage()
    sys.exit()

if (args.verbose == True):
    print "Verbose mode activated"
    verbose = 1

#======================================================== MODEL ESTIMATION MODE ========================================================#
#============================ READ FILE ============================#

if (program_mode == 1):
    print "\nLaunching {} in MODEL ESTIMATION mode".format(sys.argv[0])
    print "\n================== READING FILE =================="

    with f:
        reader = csv.reader(f)

        (num_freq_domains,default_freqs,num_components_domains) = readFirstThreeLines(reader)

        if (num_freq_domains > 2):
            print "Current version of gpmTOOL only supports model training with 1 or 2 frequency domains."
            sys.exit()

        total_num_utils = np.sum(num_components_domains, dtype=np.int32)
        print "Num total modelled components = {}".format(total_num_utils)

        # Parse the rest of the file
        P_list=np.array([], dtype = np.float32)
        F_list=[]
        U_list=[]

        #iterate for each entry
        for row_num, row in enumerate(reader):

            #check if format is correct
            if (len(row) != total_num_utils + num_freq_domains + 1):
                print "Wrong format on line {}".format(row_num+4) # data begins on line 4
                sys.exit()

            #read the power entry
            P_list = np.append(P_list, row[0])
                # print "Power: {} W".format(row[0])

            #read the frequency(ies) entry(ies)
            aux_f_row = [row[freq_id] for freq_id in range(1,num_freq_domains+1)]
            F_list.append(aux_f_row)

            #read the utilization(s) entry(ies)
            aux_u_row = [row[idx] for idx in range(num_freq_domains+1,num_freq_domains+1+total_num_utils)]
            U_list.append(aux_u_row)

        #convert the list of frequencies into an array
        F = np.asarray(F_list,  dtype=np.int32)
        U = np.asarray(U_list,  dtype=np.float32)
        P = np.asarray(P_list,  dtype=np.float32)
        # print "test1: {}".format(F[3,1])
        # print "test2: {}".format(F_aux[3][1])

    num_data_points = len(P)
    print "\nTotal data points read: {}".format(num_data_points)

    # print "\nPowers read [W]:\n {}".format(np.asarray(P,  dtype=np.float32))
    # print "\nFrequencies read [MHz]:\n {}".format(F)
    # print "\nUtilizations read [0-1]:\n {}".format(U)

    #============================ TRAINNG ALGORITHM ============================#
    print "\n\n================== TRAINING MODEL =================="
    print "\n======== STEP 1 - Initial configuration ========"

    #determine what are the frequencies going to be used in the initial training stage
    idx_other_freq = np.zeros(num_freq_domains, dtype=np.int32)
    idx_training_freqs = [None]*num_freq_domains
    read_freqs = [None]*num_freq_domains
    for domain_id in range(0, num_freq_domains):
        idx_training_freqs[domain_id]=[]
        read_freqs[domain_id] = np.unique(F[:,domain_id])
        idx_default_freq = int(np.where(read_freqs[domain_id] == default_freqs[domain_id])[0])
        idx_training_freqs[domain_id].append(idx_default_freq)

        if (idx_default_freq < len(read_freqs[domain_id])-1):
            idx_other_freq = idx_default_freq+1
        elif (idx_default_freq > 0):
            idx_other_freq = idx_default_freq-1
        else:
            print "ERROR: Not enough frequencies on domain {} to train the model".format(domain_id)
            sys.exit()

        idx_training_freqs[domain_id].append(idx_other_freq)

    # print (read_freqs)
    # if (verbose == 1):
    #     print "Index training configs:\n{}".format(idx_training_freqs)

    training_configs =  [None]*(num_freq_domains+1)
    training_configs[0] = []
    for domain_id in range(0, num_freq_domains):
        training_configs[0].append(read_freqs[domain_id][idx_training_freqs[domain_id][0]])

    for config_id in range(0, num_freq_domains):
        training_configs[config_id+1] = []
        for freq_id in range(0, num_freq_domains):
            if (freq_id == config_id):
                # print "{},{},{}".format(freq_id, idx_training_freqs[freq_id], idx_training_freqs[freq_id][1])
                training_configs[config_id+1].append(read_freqs[freq_id][idx_training_freqs[freq_id][1]])
            else:
                training_configs[config_id+1].append(read_freqs[freq_id][idx_training_freqs[freq_id][0]])

    if (verbose == 1):
        print "\nInitial training configurations:"
        print2DArray_ints(training_configs)

    training_lines = findLines_AllConfigs(training_configs)

    if (verbose == 1):
        print "\nInitial training values: {} data points".format(len(training_lines))

    P_model_begin = np.zeros(len(training_lines), dtype = np.float32)
    X_model_begin = [None]*len(training_lines);
    for data_idx,data_num in enumerate(training_lines):
        X_model_begin[data_idx] = np.ones(total_num_utils+2*num_freq_domains, dtype = np.float32)
        for domain_id in range(0, num_freq_domains):
            # print "domain_id: {}, domain_id+num_freq_domains: {}".format(domain_id, domain_id+num_freq_domains)

            X_model_begin[data_idx][domain_id] = 1 #Vc = 1 at the reference frequency configurations (initialization)
            X_model_begin[data_idx][domain_id+num_freq_domains] = 1 * F[data_num][domain_id] #Vc = 1 at the reference frequency configurations (initialization)

        idx_aux = 0
        for domain_id in range(0, num_freq_domains):
            for component_id in range(0,num_components_domains[domain_id]):
                X_model_begin[data_idx][idx_aux+2*num_freq_domains] = 1 * F[data_num][domain_id] * U[data_num][idx_aux]
                idx_aux = idx_aux + 1

        P_model_begin[data_idx] = P[data_num]

    B, rnorm = nnls(np.vstack(X_model_begin), P_model_begin)

    if (verbose == 1):
        print "\nInitial coefficient values:"
        printBCoefficients(B, num_freq_domains, num_components_domains, 1)

    # find the different possible frequency configurations (possible combinations of frequencies from each dommain)
    if (num_freq_domains == 2):
        different_F_pairs = cartesian(read_freqs)
    else:
        different_F_pairs = read_freqs[0]

    if (verbose == 1):
        print "\nDifferent F configs:"
        if (num_freq_domains == 2):
            print2DArray_ints(different_F_pairs)
        else:
            printArray_ints(different_F_pairs)

    if (num_freq_domains == 2):
        V_main = np.ones((len(read_freqs[1]), len(read_freqs[0])), dtype=np.float32)
    else:
        V_main = np.ones(len(read_freqs[0]), dtype=np.float32)

    if (verbose == 1):
        print "\nInitial V:"
        if (num_freq_domains == 2):
            printVoltage(V_main, read_freqs[0], read_freqs[1], 1)
        else:
            printArray_floats(V_main, 1)

    print "\n======== STEP 2 - Iterative Heuristic Training ========"
    print "\nTraining the model for {} iterations.".format(max_iterations)
    start = time.time()

    size_X_V = 3
    threshold_count = 0
    stop_condition = 0

    all_diffs = np.ones(max_iterations, dtype=np.float32)
    for iter_id in range(0, max_iterations):
        if (verbose == 0):
            print_progress(iter_id, max_iterations)
        else:
            print '\nIteration {}'.format(iter_id)

        # Step 2.1 - Determine the voltage for each frequency configuration
        for config_id, config in enumerate(different_F_pairs):
            # print "\nConfig: {} ({},{})".format(config, np.where(read_freqs[0]==config[0])[0][0], np.where(read_freqs[1]==config[1])[0][0])

            if (num_freq_domains == 2):
                if (config[0] == default_freqs[0]):
                    newV = 1
                else:

                    lines_config = findLines_OneConfig(config)[0]
                    # print "Lines: {}".format(len(lines_config))

                    X_model_All_V = [None]*len(lines_config);
                    P_model = np.zeros(len(lines_config), dtype=np.float32);
                    for array_id, data_idx in enumerate(lines_config):
                        # print "{},{}".format(F[data_idx][0], F[data_idx][1])
                        X_model_All_V[array_id] = np.ones(size_X_V, dtype = np.float32)

                        nc1 = num_components_domains[0]
                        nc2 = num_components_domains[1]

                        X_model_All_V[array_id][0] = B[0]
                        X_model_All_V[array_id][1] = F[data_idx][0] * (B[2] + np.sum(B[4:(4+nc1)]*(U[data_idx][0:nc1])))

                        X_model_All_V[array_id][2] = B[1] + (F[data_idx][1] * (B[3] + B[4+nc1:4+nc1+nc2]*(U[data_idx][nc1:nc1+nc2])))

                        P_model[array_id] = P[data_idx]

                    X_model_V = np.asarray(X_model_All_V, dtype=np.float32)

                    v0 = V_main[np.where(read_freqs[1]==config[1])[0][0]][np.where(read_freqs[0]==config[0])[0][0]] #previous value used as initial estimate

                    #boundaries of the voltage for the model determination
                    if (np.where(read_freqs[0]==config[0])[0][0] == 0):
                        lb = 0
                    else:
                        lb = V_main[np.where(read_freqs[1]==config[1])[0][0]][(np.where(read_freqs[0]==config[0])[0][0])-1]

                    # print "{}, {}".format(np.where(read_freqs[0]==config[0])[0][0], (len(read_freqs[0])-1))
                    if (np.where(read_freqs[0]==config[0])[0][0] == (len(read_freqs[0])-1)):
                        ub = 2
                    else:
                        ub = V_main[np.where(read_freqs[1]==config[1])[0][0]][(np.where(read_freqs[0]==config[0])[0][0])+1]

                    if (ub == lb):
                        newV = ub
                    else:
                        #determines the voltage
                        res_lsq = least_squares(fun_V, v0, args=(X_model_V[:,0], X_model_V[:,1], X_model_V[:,2], P_model), bounds=(lb, ub)) #
                        newV = res_lsq.x[0]

                    V_main[np.where(read_freqs[1]==config[1])[0][0]][np.where(read_freqs[0]==config[0])[0][0]] = newV

            else:
                if (config == default_freqs[0]):
                    newV = 1
                else:
                    lines_config = findLines_OneConfig(config)[0]
                    # print "Lines: {}".format(len(lines_config))

                    X_model_All_V = [None]*len(lines_config);
                    P_model = np.zeros(len(lines_config), dtype=np.float32);
                    for array_id, data_idx in enumerate(lines_config):
                        # print "{},{}".format(F[data_idx][0], F[data_idx][1])
                        X_model_All_V[array_id] = np.ones(size_X_V, dtype = np.float32)

                        X_model_All_V[array_id][0] = B[0]
                        X_model_All_V[array_id][1] = F[data_idx][0] * (B[1] + np.sum(B[2:(2+total_num_utils)]*(U[data_idx][0:total_num_utils])))
                        X_model_All_V[array_id][2] = 0

                        P_model[array_id] = P[data_idx]

                    X_model_V = np.asarray(X_model_All_V, dtype=np.float32)

                    v0 = V_main[np.where(read_freqs[0]==config)[0][0]] #previous value used as initial estimate

                    #boundaries of the voltage for the model determination
                    if (np.where(read_freqs[0]==config)[0][0] == 0):
                        lb = 0
                    else:
                        lb = V_main[(np.where(read_freqs[0]==config)[0][0])-1]

                    # print "{}, {}".format(np.where(read_freqs[0]==config[0])[0][0], (len(read_freqs[0])-1))
                    if (np.where(read_freqs[0]==config)[0][0] == (len(read_freqs[0])-1)):
                        ub = 2
                    else:
                        ub = V_main[(np.where(read_freqs[0]==config)[0][0])+1]

                    if (ub == lb):
                        newV = ub
                    else:
                        #determines the voltage
                        res_lsq = least_squares(fun_V, v0, args=(X_model_V[:,0], X_model_V[:,1], X_model_V[:,2], P_model), bounds=(lb, ub)) #
                        newV = res_lsq.x[0]


                    V_main[np.where(read_freqs[0]==config)[0][0]] = newV

            # print V_main

        if (verbose == 1):
            print "\tNew V:".format(iter_id+1)
            if (num_freq_domains == 2):
                printVoltage(V_main, read_freqs[0], read_freqs[1], 2)
            else:
                printArray_floats(V_main, 2)

        # Step 2.2 - Re-determine the coeffiecients B for the new voltage values
        P_model_begin = np.zeros(num_data_points, dtype = np.float32)
        X_model_begin = [None]*num_data_points
        for data_idx in range(0, num_data_points):
            X_model_begin[data_idx] = np.ones(total_num_utils+2*num_freq_domains, dtype = np.float32)

            if (num_freq_domains == 1):
                v_aux = V_main[np.where(read_freqs[0]==F[data_idx][0])[0][0]]
                X_model_begin[data_idx][0] = v_aux
                X_model_begin[data_idx][1] =  v_aux * v_aux * F[data_idx][0]
                for component_id in range(0,num_components_domains[0]):
                    X_model_begin[data_idx][component_id+2] = v_aux * v_aux * F[data_idx][0] * U[data_idx][component_id]
            else:
                for domain_id in range(0, num_freq_domains):
                    # print "domain_id: {}, domain_id+num_freq_domains: {}".format(domain_id, domain_id+num_freq_domains)
                    if domain_id == 0:
                        v_aux = V_main[np.where(read_freqs[1]==F[data_idx][1])[0][0]][np.where(read_freqs[0]==F[data_idx][0])[0][0]]
                    else:
                        v_aux = 1

                    X_model_begin[data_idx][domain_id] = v_aux #Vc = 1 at the reference frequency configurations (initialization)
                    X_model_begin[data_idx][domain_id+num_freq_domains] = v_aux * v_aux * F[data_idx][domain_id] #Vc = 1 at the reference frequency configurations (initialization)

                idx_aux = 0
                for domain_id in range(0, num_freq_domains):
                    if domain_id == 0:
                        v_aux = V_main[np.where(read_freqs[1]==F[data_idx][1])[0][0]][np.where(read_freqs[0]==F[data_idx][0])[0][0]]
                    else:
                        v_aux = 1

                    for component_id in range(0,num_components_domains[domain_id]):
                        X_model_begin[data_idx][idx_aux+2*num_freq_domains] = v_aux * v_aux * F[data_idx][domain_id] * U[data_idx][idx_aux]
                        idx_aux = idx_aux + 1

            P_model_begin[data_idx] = P[data_idx]

        oldB = B
        B, rnorm = nnls(np.vstack(X_model_begin), P_model_begin)

        diff_Bs_aux = np.zeros(len(B), dtype=np.float32)
        for i, value in enumerate(B):
            if (oldB[i] != 0):
                diff_Bs_aux[i] = abs((oldB[i] - B[i]) / oldB[i])
        diff_Bs = max(diff_Bs_aux)*100
        all_diffs[iter_id] = diff_Bs

        if (verbose == 1):
            print "\n\tNew coefficient values ({:6.3f}% difference):".format(diff_Bs)
            printBCoefficients(B, num_freq_domains, num_components_domains, 2)

        if (diff_Bs < threshold):
            threshold_count+=1
        else:
            threshold_count=0

        if (threshold_count == 5):
            stop_condition = 1
            break


    #============================ PRESENT FINAL MODEL ============================#
    end = time.time()
    if (stop_condition == 0):
        print "\n\n======== FINISHED MODEL DETERMINATION - Max Iteration reached ========"
        if diff_Bs > 1:
            print "\n====== IMPORTANT: Difference between last two iterations: {:6.3f}%. Please consider increasing the number of iterations. ======".format(diff_Bs)
    else:
        print "\n\n======== FINISHED MODEL DETERMINATION - Convergence achieved in {} iterations ========".format(iter_id)

    print "Training duration: {:.2f} s".format(end-start)

    print "\n\nFinal model coefficient values:"
    printBCoefficients(B, num_freq_domains, num_components_domains, 1)

    print "\nFinal voltage values:"
    if (num_freq_domains == 2):
        printVoltage(V_main, read_freqs[0], read_freqs[1], 1)
    else:
        printArray_floats(V_main, 1)

    #check if values are to be written in output file
    if (args.output_file != None):
        #current version only prints model if num_freq_domains=2
        if (num_freq_domains != 2):
            print "Current version of gpmTOOL can only save model file if number of frequency domains is 2."
        else:
            try:
                f = open(args.output_file, 'wb')
            except:
                print "Could not read file:", args.output_file
                sys.exit()

            print "\nWriting values in output file: {}".format(args.output_file)

            with f:
                writer = csv.writer(f)

                #first 3 lines with the device information
                writer.writerow([num_freq_domains])
                writer.writerow([freq for freq in default_freqs])
                writer.writerow([num_comp for num_comp in num_components_domains])

                #write the different frequencies read (required to know which F config the voltage values correspond to)
                for domain_id in range(0,num_freq_domains):
                    writer.writerow(read_freqs[domain_id])

                writer.writerow(B)
                writer.writerows(V_main)

    #display convergence rate between iterations
    plt.figure(1)
    plt.title('Convergence achieved in {} iterations ({:.2f}s)'.format(iter_id, end-start))
    plt.xlabel('Iteration number')
    plt.ylabel('Difference to previous iter (%)')
    plt.plot(all_diffs[0:iter_id])

    #display determined voltage values
    plt.figure(2)
    if (num_freq_domains == 2):
        for freq_mem_id, freq_mem in enumerate(read_freqs[1]):
            txt_aux = "Fmem = {} MHz".format(freq_mem)
            plt.plot(read_freqs[0], V_main[freq_mem_id], label=txt_aux)
    else:
        plt.plot(read_freqs[0], V_main)


    plt.axis([read_freqs[0][0]//100*100, np.ceil(read_freqs[0][len(read_freqs[0])-1] / 100.0)*100, 0, ymax_V_plot])
    plt.grid(True)
    plt.xlabel('Core Frequency [MHz]')
    plt.ylabel('Core Voltage / Reference Voltage')
    if (num_freq_domains == 2):
        plt.legend(loc=0)
    plt.title('Determined voltage values:')

    plt.show()

#======================================================== POWER PREDICTION MODE ========================================================#
else:
    print "\nLaunching {} in Power Consumption Prediction mode".format(sys.argv[0])

    #============================ READ MODEL FILE ============================#
    with f1:
        print "\n================== READING MODEL FILE =================="
        reader = csv.reader(f1)

        (num_freq_domains,default_freqs,num_components_domains) = readFirstThreeLines(reader)
        total_num_utils = np.sum(num_components_domains, dtype=np.int32)

        read_freqs = [None]*num_freq_domains

        if (read_freqs != 2):
            print ("Error: Current version only supports predictions when number of frequency domains is equal to 2.")
            sys.exit()

        for domain_id in range(0, num_freq_domains):
            line = (next(reader))
            read_freqs[domain_id] = np.zeros(len(line), dtype=np.int32)
            for i,value in enumerate(line):
                read_freqs[domain_id][i]=value

        # find the different possible frequency configurations (possible combinations of frequencies from each dommain)
        different_F_pairs = cartesian(read_freqs)


        if (verbose == 1):
            print "\nDifferent F configs:"
            print2DArray_ints(different_F_pairs)

        # Parse the fourth line with values of the coefficients
        sizeB = (2*num_freq_domains+total_num_utils)
        line_B = (next(reader))
        if len(line_B) != sizeB:
            print "\nWrong format on line {} of file {}".format(4+num_freq_domains, f1.name)
            print "len(line): {} != 2*num_freq_domains+total_num_utils: {} ".format(len(line_B), sizeB)
            sys.exit()
        B = np.zeros(sizeB, dtype=np.float32)
        for i,value in enumerate(line_B):
            B[i] = float(value)

        #Parse the remaining lines with the values of V
        V=np.zeros((len(read_freqs[1]), len(read_freqs[0])), dtype = np.float32)

        #iterate for each entry
        for row_num, row in enumerate(reader):

            if (row_num >= len(read_freqs[1])):
                print("\nToo many lines in file {}".format(f1.name))
                print("({} voltage rows are given when number of mem frequencies is only {})".format(row_num+1, len(read_freqs[1])))

                sys.exit()

            #check if format is correct
            if (len(row) != len(read_freqs[0])):
                print "\nWrong format on line {}".format(row_num+5+num_freq_domains)
                sys.exit()

            for i,value in enumerate(row):
                V[row_num][i] = float(value)

    print "\nCoefficient values read from file {}:".format(f1.name)
    printBCoefficients(B, num_freq_domains, num_components_domains, 1)

    print "\nVoltage values read from file {}:".format(f1.name)
    if (num_freq_domains == 2):
        printVoltage(V, read_freqs[0], read_freqs[1], 1)
    else:
        print2DArray_floats(V, 1)

    #============================ READ FILE WITH BENCHMARKS TO MAKE PREDICTIONS ============================#
    with f2:
        print "\n=============== READING BENCHMARKS FILE ================"
        reader = csv.reader(f2)

        (num_freq_domains_2,default_freqs_2,num_components_domains_2) = readFirstThreeLines(reader)

        #verify that both files are considering the same device characteristics
        if (num_freq_domains != num_freq_domains_2):
            print "\nFiles do not match: {}.num_freq_domains != {}.num_freq_domains".format(f1.name,f2.name)
            print "{} != {}".format(num_freq_domains, num_freq_domains_2)
            sys.exit()
        else:
            for domain_id in range(0, num_freq_domains):
                if (default_freqs[domain_id] != default_freqs_2[domain_id]):
                    print "\nFiles do not match: {}.default_freqs[domain_id={}] != {}.default_freqs.[domain_id={}]".format(f1.name,domain_id,f2.name,domain_id)
                    print "{} != {}".format(default_freqs[domain_id], default_freqs_2[domain_id])
                    sys.exit()
                elif (num_components_domains[domain_id] != num_components_domains_2[domain_id]):
                    print "\nFiles do not match: {}.num_components_domains[domain_id={}] != {}.num_components_domains_2.[domain_id={}]".format(f1.name,domain_id,f2.name,domain_id)
                    print "{} != {}".format(num_components_domains[domain_id], num_components_domains_2[domain_id])
                    sys.exit()

        #iterate for each entry
        utils = []
        for row_num, row in enumerate(reader):
            if (len(row) != total_num_utils):
                print "\nWrong number of utilizations on line {} of file {} (expected {} utilization values).".format(row_num+4, f2.name, total_num_utils)
                sys.exit()
            utils.append(row)



    num_benchs = len(utils)
    utils = np.asarray(utils,dtype=np.float32)
    if (verbose == 1):
        print "\nUtilizations from {} benchmarks read from file {}:".format(num_benchs, f2.name)
        print2DArray_floats(utils,1)
    else:
        print "\nSuccessfully read {} benchmarks from file {}".format(num_benchs, f2.name)


    #============================ MAKE POWER BREAKDOWN PREDICTION ============================#
    print "\n=============== ESTIMATING POWER BREAKDOWN ================"

    cm = plt.get_cmap('gist_rainbow')

    if (total_num_utils+2 < 20):
        colors = tableau20[::2]
    else:
        colors = [cm(1.*i/(total_num_utils+2)) for i in range((total_num_utils+2))]

    # colors = [cm(1.*i/(total_num_utils+2)) for i in range((total_num_utils+2))]

    bar_l = np.arange(1, num_benchs+1)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    plt.figure(1, facecolor='white')

    #note that at the reference the voltage is 1

    P_breakdown = [None]*num_benchs
    list_Names = []

    #compute the estimative of each component of the power
    for bench_id in range(0,num_benchs):
        P_breakdown[bench_id] = np.zeros(total_num_utils+1, dtype = np.float32)

        list_Names.append("B"+str(bench_id+1))

        idx_aux=0
        for domain_id in range(0, num_freq_domains):
            P_breakdown[bench_id][0] = P_breakdown[bench_id][0] + B[domain_id]*1 + B[domain_id+num_freq_domains]*(1**2)*default_freqs[domain_id] #constant power of the frequency configuration

            for component_id in range(0,num_components_domains[domain_id]):
                P_breakdown[bench_id][idx_aux+1] = B[idx_aux+2*num_freq_domains]*(1**2)*default_freqs[domain_id]*utils[bench_id][idx_aux]
                idx_aux += 1

    P_breakdown = np.array(P_breakdown)

    print "\nDefault frequency configuration:"
    for domain_id in range(0,num_freq_domains):
        print "\tFreq_domain_{}: {} MHz".format(domain_id, default_freqs[domain_id])

    # if (verbose == 1):
    print "\nConstant power at default frequency configuration: {:5.1f}W".format(P_breakdown[0,0])

    #total power estimative
    total_P = np.sum(P_breakdown, axis=1)

    #plot the bar with the constant values
    p0 = plt.bar(bar_l, P_breakdown[:,0], bar_width, label='Pconstant', color=colors[0])

    Pbottom = P_breakdown[:,0]
    p1 = [None]*total_num_utils

    #plot the bars with the dynamic values (dependent to the Utilization values read)
    idx_aux=0
    for domain_id in range(0, num_freq_domains):
        for component_id in range(0,num_components_domains[domain_id]):
            txt_aux = 'Pdyn_domain{}_component{}'.format(domain_id,component_id)
            p1[idx_aux] = plt.bar(bar_l, P_breakdown[:,idx_aux+1], bar_width, bottom=Pbottom, label=txt_aux,color=colors[idx_aux+1])
            Pbottom = Pbottom + P_breakdown[:,idx_aux+1]
            idx_aux += 1

    #Loop to print the percentage values in the plot
    for bench_id in range(0, num_benchs):
        rect = p0[bench_id]
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()

        label = "{:3.1f}%".format(P_breakdown[bench_id,0] / total_P[bench_id] * 100)
        plt.text(x + bar_width/2., ((y+bar_min_Y) + (height-bar_min_Y)/2.), label, ha='center', va='center', fontsize=8)

        idx_aux = 0
        for domain_id in range(0, num_freq_domains):
            for component_id in range(0,num_components_domains[domain_id]):
                rect = p1[idx_aux][bench_id]
                # w = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                height = rect.get_height()

                if (P_breakdown[bench_id,idx_aux+1] / total_P[bench_id] * 100 > bar_print_threshold):
                    label = "{:3.1f}%".format(P_breakdown[bench_id,idx_aux+1] / total_P[bench_id] * 100)
                    plt.text(x + bar_width/2., y + height/2., label, ha='center', va='center', fontsize=8)

                idx_aux += 1

        #print the total power consumption on top of the bars
        plt.text(x + bar_width/2., total_P[bench_id]+2, "{:3.0f}W".format(total_P[bench_id]), ha='center', va='center', fontsize=10, fontweight='bold')

    plt.title("Power Breakdown at the default frequency configuration")
    plt.ylim([bar_min_Y,max_P])
    plt.ylabel("Predicted Power [W]", fontsize=18)
    plt.xlabel('Benchmarks', fontsize=18)
    plt.legend(loc=0)

    # print list_Names
    plt.xticks(tick_pos, list_Names, fontsize=13)
    plt.xlim([0,num_benchs+1])
    plt.yticks(fontsize=14)


    print "\nPower consumption breakdown at the default frequency configuration:"
    printPowerBreakdown(P_breakdown)


    #============================ MAKE POWER DVFS PREDICTION ============================#
    #only working with num_freq_domains == 2
    if (num_freq_domains == 2):
        print "\n=============== ESTIMATING POWER VARIATIONS WITH DVFS ================"

        print "\nDVFS Power Consumption Predictions:\n"

        plot_nrows = int(np.ceil(np.ceil(num_benchs/benchs_per_row)))
        # plt.figure(2)
        fig2, axs = plt.subplots(nrows=plot_nrows,ncols=benchs_per_row)
        fig2.suptitle('DVFS Power Consumption Prediction')

        for bench_id in range(0, num_benchs):
            idx_row = int(np.floor(bench_id/benchs_per_row))
            idx_col = bench_id % benchs_per_row

            print "Power Benchmark {}:".format(bench_id+1)

            #print fcore values (header)
            s = "\t{:<3}Fcore [MHz]:".format('')
            for clock_id, clock in enumerate(read_freqs[0]):
                if clock_id > 0:
                    s+=', '
                s += "{:5d}".format(clock)
            print s

            for clock_mem_id, clock_mem in enumerate(read_freqs[1]):
                vm = 1
                s = "\tFmem={:4d}MHz: [".format(clock_mem)

                P=np.zeros(len(read_freqs[0]), dtype=np.float32)
                for clock_id, clock_core in enumerate(read_freqs[0]):
                    vc = V[clock_mem_id, clock_id]
                    P[clock_id] = B[0]*vc +B[1]*vm + B[2]*vc**2*clock_core + B[3]*vm**2*clock_mem

                    idx_aux = 0
                    for domain_id in range(0, num_freq_domains):
                        if (domain_id == 0):
                            f = clock_core
                            v = vc
                        else:
                            v = vm
                            f = clock_mem
                        for component_id in range(0, num_components_domains[domain_id]):
                            P[clock_id] = P[clock_id] + B[2*num_freq_domains+idx_aux]*v**2*f*utils[bench_id][idx_aux]
                            idx_aux += 1

                    if (clock_id > 0):
                        s += ', '
                    s += "{:5.1f}".format(P[clock_id])

                s += '] W'
                print s

                axs[idx_row][idx_col].grid(True)

                if (idx_row < plot_nrows-1):
                    axs[idx_row][idx_col].xaxis.set_ticklabels([])
                else:
                    axs[idx_row][idx_col].set_xlabel('Core Frequency [MHz]')
                if (idx_col > 0):
                    axs[idx_row][idx_col].yaxis.set_ticklabels([])
                else:
                    axs[idx_row][idx_col].set_ylabel('Power [W]')

                txt_aux = "Fmem = {} MHz".format(clock_mem)
                axs[idx_row][idx_col].plot(read_freqs[0], P, label=txt_aux)

            axs[idx_row][idx_col].axis([read_freqs[0][0]//100*100, np.ceil(read_freqs[0][len(read_freqs[0])-1] / 100.0)*100, 0, max_P])
            axs[idx_row][idx_col].set_title("B{}".format(bench_id+1))

        plt.legend(loc = 'upper center', bbox_to_anchor = (0,-0.04,1,1), ncol=4, bbox_transform = plt.gcf().transFigure )

    plt.show()
