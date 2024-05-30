import numpy as np
import math
import scipy
import torch 


# Akinesia : want some sort of metric for how long movement initialization is delayed
# Returns the average delay over trial per muscle from 'test' data compared to 'base' data, 
#       using the hold signals as specified in input
#       Also returns some measure of its statistical significance
#
#       input is shape [numtrials][1][length of each trial]
#       test and base should have shape [numtrials][50][length of each trial]

def akinesia(test, base, input_test, epsilon = 0.05):

    # loop over the test trials creating all_test_delays, a trials x 50 matrix with the delay from each muscle during each trial
    num_trials_test = input_test.shape[0]
    all_test_delays = np.zeros((num_trials_test, 50))
    for ind,trial in enumerate(input_test):
        inxs = np.argwhere(trial<1)
        go_cue = inxs[0]
        print(trial.shape)
        # 50 x 200 smth trial
        curr_test_trial = test[ind]
        muscle_mvmts_test = np.zeros((50))
        for i in range(curr_test_trial.shape[1]):
            muscle = curr_test_trial[:,i]
            # for each mucle, what is the first time that it is non-zero
            mvms = np.argwhere(abs(muscle)>epsilon)
            print(mvms)
            np_ver = mvms.detach().cpu().numpy()
            if np_ver[0].size != 0:
                muscle_mvmts_test[i] = mvms[0][0]
        # subtract that index from the go_cue !! NEEDS TO BE CHANGED TO GO CUE FROM IMAGE !!
        all_test_delays[ind] = muscle_mvmts_test - go_cue

    # same thing as above for base trials
    num_trials_base = input_test.shape[0]
    all_base_delays = np.zeros((num_trials_base, 50))
    for ind,trial in enumerate(input_test):
        inxs = np.argwhere(trial<1)
        go_cue = inxs[0]

        # 50 x 200 smth trial
        curr_base_trial = base[ind]
        muscle_mvmts_base = np.zeros((50))
        for i in range(curr_base_trial.shape[1]):
            muscle = curr_base_trial[:, i]
            mvms = np.argwhere(abs(muscle)>epsilon)
            if mvms.size != 0:
                muscle_mvmts_base[i] = mvms[0][0]

        all_base_delays[ind] = muscle_mvmts_base - go_cue
    
    # takes the average of the delays over the trials axis for both test and base, 
    # then subtracts the difference for a comparitive index
    # 1 x 50

    # 2-sample t-score --> p-value test
    # What's the probability that these samples came from the same distribution?
    mean_dist_base = np.average(all_base_delays, axis = 0)
    std_dist_base = scipy.stats.tstd(all_base_delays, axis = 0)
    mean_dist_test = np.average(all_test_delays, axis = 0)
    std_dist_test = scipy.stats.tstd(all_test_delays, axis = 0)

    t_score = (mean_dist_base - mean_dist_test) / np.sqrt((std_dist_base**2 / num_trials_base) + (std_dist_test**2 / num_trials_test))
    df = num_trials_base + num_trials_test - 2
    p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_score), df))
    
    return mean_dist_base, mean_dist_test, p_value

        
# Hypokinesia : movement amplitude, take the integral under the velocity for ‘distance traveled’
#       Direct comparison of the overall movement amplitude (sum not weighted) between two samples
#       first each sample set is looped over and the sum of the velocities for each trial is added to 
#           an array. (summed over time, so the result is 50 x 1)
#       then, the average is taken over each trial (result is 50 x 1)
#       We return the difference between the averages for test and base (50 x 1)
#       2 sample p value is calculated, 

#       input arrays are [num_trials] [50 x length of trial]
#`      num trials for test and base can be different

# Returns:
#       average distance for both base and test, and the probability that both samples were taken from the same distribution

def hypokinesia(test, base):

    # loop through all the trials in 'base', sum up the velocities across all muscles, still keeping them seperate
    # store in all_trial_int_base a trials x 50 matrix
    trials_base = base
    num_trials_base = len(base)
    all_trial_int_base = np.zeros((num_trials_base, 50))
    for ind,trial in enumerate(trials_base):
        all_trial_int_base[ind] = torch.sum(trial, axis = 0)
    
    # same thing as above but for 'test' 
    trials_test = test
    num_trials_test = len(test)
    all_trial_int_test = np.zeros((num_trials_test, 50))
    for ind,trial in enumerate(trials_test):
        all_trial_int_test[ind] = torch.sum(trial, axis = 0)
    
    # average both across trials, 1 x 50
    avg_dist_base = np.average(all_trial_int_base, axis = 0)
    avg_dist_test = np.average(all_trial_int_test, axis = 0)

    # 2-sample t-score --> p-value test
    # What's the probability that these samples came from the same distribution?
    mean_dist_base = np.sum(avg_dist_base)
    std_dist_base = scipy.stats.tstd(np.sum(all_trial_int_base, axis = 1))
    mean_dist_test = np.sum(avg_dist_test)
    std_dist_test = scipy.stats.tstd(np.sum(all_trial_int_test, axis = 1))

    t_score = (mean_dist_base - mean_dist_test) / np.sqrt((std_dist_base**2 / num_trials_base) + (std_dist_test**2 / num_trials_test))
    df = num_trials_base + num_trials_test - 2
    p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_score), df))
    
    return avg_dist_test, avg_dist_base, p_value
    
# Bradykinesia: lower curve or hesitation in the velocity
#       lower curve can be tested by comparing the maximum velocities
#       hesitation, not sure, might have to do with oscillations in the velocity 
#           could also test how quickly they get to max velocity?


def bradykinesia(test, base):

    # loop through the trails for 'base', taking the maximum velocity and storing it
    # in all_trial_max_base, trials x 50
    num_trials_base = len(base)
    all_trial_max_base = np.zeros((num_trials_base, 50))
    
    for ind,trial in enumerate(base):
        all_trial_max_base[ind] = torch.max(trial, axis = 0)[0]

    # same thing as above but for 'test'
    num_trials_test = len(test)
    all_trial_max_test = np.zeros((num_trials_test, 50))
    
    for ind,trial in enumerate(test):
        all_trial_max_test[ind] = torch.max(trial, axis = 0)[0]

    # take the average over all trials, 1 x 50
    avg_max_base = np.average(all_trial_max_base, axis = 0)
    avg_max_test = np.average(all_trial_max_test, axis = 0)    

    std_dist_base = scipy.stats.tstd(all_trial_max_base, axis = 0)
    std_dist_test = scipy.stats.tstd(all_trial_max_test, axis = 0)
    
     # 2-sample t-score --> p-value test
    # What's the probability that these samples came from the same distribution?
    t_score = (avg_max_base - avg_max_test) / np.sqrt((std_dist_base**2 / num_trials_base) + (std_dist_test**2 / num_trials_test))
    df = num_trials_base + num_trials_test - 2
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_score), df))

    return avg_max_base, avg_max_test, p_values

# Tremor Factor: It is defined as the root mean square value of the acceleration of the arm during the reaching task. 
# this actually isn't finished I don't think...

def tremor_factor(test, base):
    base_acc = vel_to_acc(base)
    num_base_trials = len(base_acc)
    
    hold_arr = np.zeros((num_base_trials, 50))
    #Calculate square
    for i,trial in enumerate(base_acc):
        for j in range(trial.shape[1]):
            muscle = trial[:, j]
            square = 0
            mean = 0.0
            root = 0.0
            for time_point in muscle:
                square += time_point**2
     
            #Calculate Mean 
            mean = (square / (float)(muscle.shape[0]))
     
            #Calculate Root
            root = math.sqrt(mean)
            hold_arr[i][j] = root
    
    avgd_base = np.average(hold_arr, axis = 0)

    test_acc = vel_to_acc(test)
    num_test_trials = len(test_acc)
    
    hold_arr_test = np.zeros((num_test_trials, 50))
    #Calculate square
    for i,trial in enumerate(test_acc):
        for j in range(trial.shape[1]):
            muscle = trial[:,j]
            square = 0
            mean = 0.0
            root = 0.0
            for time_point in muscle:
                square += time_point**2
     
            #Calculate Mean 
            mean = (square / (float)(muscle.shape[0]))
     
            #Calculate Root
            root = math.sqrt(mean)
            hold_arr_test[i][j] = root
    
    avgd_test = np.average(hold_arr_test, axis = 0) 

    std_base = scipy.stats.tstd(hold_arr, axis = 0)
    std_test = scipy.stats.tstd(hold_arr_test, axis = 0)
    
     # 2-sample t-score --> p-value test
    # What's the probability that these samples came from the same distribution?
    t_score = (avgd_base - avgd_test) / np.sqrt((std_base**2 / num_base_trials) + (std_test**2 / num_test_trials))
    df = num_base_trials + num_test_trials - 2
    p_values = 2 * (1 - scipy.stats.t.cdf(np.abs(t_score), df))
      
    return avgd_base, avgd_test, p_values


def vel_to_acc(data):
    accs = []
    
    for trial in enumerate(data):
        accs.append(np.diff(trial[1]))
    
    return accs