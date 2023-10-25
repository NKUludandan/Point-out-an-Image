## Student t-test to test significance
import numpy as np
from scipy import stats

# top-1-recall:
text1_r = [0.0935, 0.0850, 0.0881, 0.0909, 0.0883]
trace1_r = [0.1271, 0.1400, 0.1204, 0.1125, 0.1299]
# calc avg:
text1_avg = np.mean(text1_r)
trace1_avg = np.mean(trace1_r)
# calc standard deviation:
text1_sd = np.std(text1_r)
trace1_sd = np.std(trace1_r)

print("text sd: %f, text_avg: %f " % (text1_sd, text1_avg))
print("trace sd: %f, trace_avg: %f " % (trace1_sd, trace1_avg))
print()

def t_testRes(a,b):
    p = 0.001
    # ttest
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    print('t-statistic: ', t_stat)
    print('p-val: ', p_val)
    if p_val < p:
        print("Reject the null hypothesis.")
    else:
        print(print("Fail to reject the null hypothesis."))
    return

t_testRes(text1_r,trace1_r)

# top-5-recall:
text5_r = [0.3515, 0.3640, 0.3684, 0.3694, 0.3746]
trace5_r = [0.4320, 0.4233, 0.4241, 0.4351, 0.4086]

print("text sd: %f, text_avg: %f " % (np.std(text5_r), np.mean(text5_r)))
print("trace sd: %f, trace_avg: %f " % (np.std(trace5_r), np.mean(trace5_r)))
print()

t_testRes(text5_r,trace5_r)

# top-10-recall:
text10_r = [0.5503, 0.5612, 0.5588, 0.5553, 0.5503]
trace10_r = [0.6181, 0.6279, 0.6176, 0.6146, 0.6223]

print("text sd: %f, text_avg: %f " % (np.std(text10_r), np.mean(text10_r)))
print("trace sd: %f, trace_avg: %f " % (np.std(trace10_r), np.mean(trace10_r)))
print()

t_testRes(text10_r,trace10_r)



