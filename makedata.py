import pandas as pd
import numpy as np
import glob
import csv

memory_height = 5
memory_width = 13
gen_model = 2

def map_ip_str_to_int_list(ip_str, ipspace=None):
    ip_group = ip_str.split('.')
    rt = []
    pw = 1
    for i in list(reversed(range(len(ip_group)))):
        # print(i, int(ip_group[i]), pw)
        # rt += int(ip_group[i]) * pw
        rt.append(int(ip_group[i]))
        # if i>0:
        # pw *= 255
    # print(rt)
    if ipspace is not None:
        for i in range(len(rt)):
            rt[i] /= ipspace
    # print(rt)
    return rt

def transform_1_df(df, file_name):
    buffer = [[0]*memory_width] * memory_height
    output_sample = []

    df['log_byt'] = np.log(df['byt'])
    if gen_model == 1:
        bytmax = 13.249034794106116 #20.12915933105231 # df['log_byt'].max()
        teTmax = 23 # df['teT'].max()
        teDeltamax = 222# 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535
    else:
        bytmax = 20.12915933105231 # df['log_byt'].max()
        teTmax = 23 # df['teT'].max()
        teDeltamax = 1336 # df['teDelta'].max()
        ipspace = 255
        portspace = 65535 
    # print('bytmax', bytmax, 'teTmax', teTmax, 'teDeltamax', teDeltamax, 'ipspace', ipspace)

    # make (n - 50) * 50 * 10 sample
    for index, row in df.iterrows():
        # print(index)
        # print(row)
        line = [row['teT']/teTmax, row['teDelta']/teDeltamax, row['log_byt']/bytmax]
        line += map_ip_str_to_int_list(row['sa'], ipspace)
        line += map_ip_str_to_int_list(row['da'], ipspace)
        line += [row['sp']/portspace, row['dp']/portspace]
        
        line_pr = []
        if row['pr'] == 'TCP':
            line_pr = [1, 0, 0]
        elif row['pr'] == 'UDP':
            line_pr = [0, 1, 0]
        else:
            line_pr = [0, 0, 1]
        # line += line_pr

        # line = [row['teT'], row['teDelta'], row['log_byt']]
        # line += map_ip_str_to_int_list(row['sa'])
        # line += map_ip_str_to_int_list(row['da'])
        # line += [row['sp'], row['dp']] 

        # print(len(line), line)
        # input()
        buffer.append(line)
        # if len(buffer)<51:
            # continue
        line_with_window = []
        for l in buffer[-memory_height-1:]:
            line_with_window += l
        # line_with_window = buffer[-50:]
        # print(line_with_window)
        # print(len(line_with_window))
        # input()
        output_sample.append(line_with_window)
    # print('linewithwindow', line_with_window)
    print(len(line_with_window), len(output_sample))

    with open(file_name,'a') as f:
        writer = csv.writer(f)
        writer.writerows(output_sample)

print('========================start makedata==================')
if gen_model == 1:
    with open('train.csv','w') as f:
        pass
    source_data = 'expanded_day_1_42.219.145.151.csv'
    df = pd.read_csv(source_data)
    transform_1_df(df, 'train.csv')
else:
    # df = pd.concat([pd.read_csv(f) for f in glob.glob('cleaned_data/*.csv')], ignore_index=True)
    with open('all_train.csv','w') as f:
        pass
    with open('all_test.csv', 'w') as f:
        pass
    count = 0
    for f in glob.glob('cleaned_data/*.csv'):
        print('making for', f)
        df = pd.read_csv(f)
        if count < 90:
            transform_1_df(df, 'all_train.csv')
        else:
            transform_1_df(df, 'all_test.csv')
        count += 1


#TODO: normalization to fixed range
#TODO: append the autoencoder module

# import csv
# with open('all_train.csv','w') as f:
#     writer = csv.writer(f)
#     writer.writerows(output_sample)

# with open('test.csv','w') as f:
#     writer = csv.writer(f)
#     writer.writerows(output_sample)