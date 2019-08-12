import torch.nn as nn
import torch.optim as optim
import mdn
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import torch
import csv
import time

# initialize config
skip_train = False
EPOCH = 300
train_model = 2
if train_model == 1:
    bytmax = 13.249034794106116
    teDeltamax = 222
    train_file = 'train.csv'
    test_file = 'train.csv'
    gen_file = 'gen.csv'
else:
    bytmax = 20.12915933105231
    teDeltamax = 1336
    train_file = 'all_train.csv'
    test_file = 'all_test.csv'
    gen_file = 'all_gen.csv'
traffic_feature_num = 13
memory_height = 5
conv_num = 10

start_time = time.time()

# initialize the model
print('='*25+'start'+'='*25)
model = mdn.MDN(memory_height, traffic_feature_num, conv_num, traffic_feature_num, 7)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.9)

from torchsummary import summary
summary(model, (memory_height, traffic_feature_num))

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        single_image_label = np.asarray(self.data.iloc[index]).reshape(self.height,self.width).astype(np.float32)[-1]
        img_as_np = np.asarray(self.data.iloc[index]).reshape(self.height,self.width).astype(np.float32)[:-1]
        img_as_tensor = torch.from_numpy(img_as_np)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

# prepare the data
train_from_csv = CustomDatasetFromCSV(train_file, memory_height+1, traffic_feature_num)
train_loader = torch.utils.data.DataLoader(dataset=train_from_csv, batch_size=64, shuffle=True)

test_from_csv = CustomDatasetFromCSV(test_file, memory_height+1, traffic_feature_num)

vali_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_from_csv, batch_size=1, shuffle=False)

def test_validation():
    for step, (x, b_label) in enumerate(vali_loader):
        minibatch = x.view(-1, memory_height, traffic_feature_num)
        pi, sigma, mu = model(minibatch)
        loss = mdn.mdn_loss(pi, sigma, mu, b_label) 
        return loss

# train the model
if skip_train == True:
    model.load_state_dict(torch.load('./saved_model/conv1d_model1.pkl'))
else:
    print('='*25+'start training'+'='*25) 
    train_loss = []
    test_loss = []
    model.train()
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            minibatch = x.view(-1, memory_height, traffic_feature_num)
            model.zero_grad()
            pi, sigma, mu = model(minibatch)
            loss = mdn.mdn_loss(pi, sigma, mu, b_label)
            # if np.isnan(pi.data.numpy()[0][0]):
            #     print('pi', pi)
            #     print('sigma', sigma)
            #     print('mu', mu)
            #     print(minibatch)
            #     input()
            
            # print(train_loss)
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                validation_loss = test_validation()

                print('Epoch: ', epoch, '| train loss: %.4f | validation loss: %.4f' % (loss.data.numpy(), validation_loss.data.numpy()))
                train_loss.append([epoch, loss.data.tolist(), validation_loss.data.numpy()])
        scheduler.step()
        print('current lr', optimizer.state_dict()['param_groups'][0]['lr'])
                
    with open('loss.csv','w') as f:
        writer = csv.writer(f)
        writer.writerows(train_loss)

    torch.save(model.state_dict(), './saved_model/conv1d_model1.pkl')

def deal_with_output(samples):
    new_row = samples.data.numpy()[-1]
    # print('new_row', new_row)
    # print(samples.size(), 'newrow', new_row)
    # print(new_row[2])
    l_np = np.asarray(new_row[-3:])
    protocol = [0, 0, 0]
    # print(new_row[7:10])
    # print(l_np)
    protocol[l_np.argmax()] = 1
    # print('l_np', l_np, protocol)

    
    final_new_row = [int(new_row[0]), int(new_row[1] * teDeltamax), int(np.ceil(np.exp(new_row[2] * bytmax)))]
    for i in range(3, 3+8):
        final_new_row.append(int(new_row[i]) * 255)
    final_new_row += [int(new_row[11]*65535),int(new_row[12]*65535)]
    # final_new_row += protocol

    print(len(final_new_row))
    return final_new_row
    # print('new_new_row', new_row)
    
    # print('marginal', marginal.size())
    # print('marginal0', marginal[0].size())
    # print('samples', samples.size())
    # print('samples0', samples[0].size())

# sample new points from the trained model
print('='*25+'start testing'+'='*25)
marginal = None
generated_rows = []
i = 0
for step, (x, b_label) in enumerate(test_loader):
    # print('testing:::::::', x.size())
    marginal = x.view(-1, memory_height*traffic_feature_num)[0]
    break
    minibatch = x.view(-1, memory_height, traffic_feature_num)
    # print(minibatch)
    model.zero_grad()
    pi, sigma, mu = model(minibatch)
    samples = mdn.sample(pi, sigma, mu)
    samples[-1][0] = torch.round(minibatch[0][-1][0] * 23)
    # print('minihour', minibatch[-1][0][0])
    new_row = deal_with_output(samples)
    generated_rows.append(new_row)
    i+=1
    print('%dth row'%i, new_row)

# import csv
# with open('gen_fromreal.csv','w') as f:
#     writer = csv.writer(f)
#     writer.writerows(generated_rows)

# exit()

# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))
tot_time = 0
i = 0
generated_rows = []
print('='*25+'testing'+'='*25)
model.eval()
# print('marginal testing', marginal.size(), marginal.view(-1, 50, 10))
while tot_time < 86400:
    # print('marginal size:', marginal.size())
    marginal = marginal.view(-1, memory_height, traffic_feature_num)
    # print(marginal.size())
    pi, sigma, mu = model(marginal)
    # print(mu)
    samples = mdn.sample(pi, sigma, mu)
    # print(samples)
    # print('samples:', samples.size())
    # print(samples)
    samples[-1][0] = tot_time/3600
    if samples[-1][1] < 0:
        samples[-1][1] = 1

    new_row = deal_with_output(samples)
    tot_time += new_row[1]
    i += 1
    
    marginal = torch.cat((marginal[0],samples), 0)
    # print('new_margin', marginal.size())
    marginal = marginal[1:]
    # print('after_margin', marginal.size())
    
    print(i, tot_time, '==>', new_row)
    generated_rows.append(new_row)

# print('test', marginal.size())
# pi, sigma, mu = model(marginal)
# samples = mdn.sample(pi, sigma, mu)
# print(samples.size())
with open(gen_file,'w') as f:
    writer = csv.writer(f)
    writer.writerows([['te', 'delta_t', 'byt', 'sa1', 'sa2', 'sa3', 'sa4', 'da1', 'da2', 'da3', 'da4', 'sp', 'dp']]) #, 'tcp', 'udp', 'other']])
    writer.writerows(generated_rows)

end_time = time.time()
print('runtime as second', end_time-start_time)