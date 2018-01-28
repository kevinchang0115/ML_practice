import csv 
import numpy as np
import math

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

# read training data
n_row = 0
text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

x = []
y = []
feature_list = [7,8,9]
#feature_list = range(18)
hour = 9
# 每 12 個月
for i in range(12):
    # 一個月取連續 hour 小時的data可以有 480-hour 筆
    for j in range(480-hour):
        x.append([])
        # n種污染物 in feature list
        for t in feature_list:
            # 連續 hour 小時
            for s in range(hour):
                x[(480-hour)*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+hour])
x = np.array(x)
y = np.array(y)

# add square term
x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 0.00001
repeat = 1000000

x_t = x.transpose()
m = np.zeros(len(x[0]))
v = np.zeros(len(x[0]))
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
old_cost = float('Inf')
ld = 0
for i in range(repeat):
    # calculate lost and cost (RMSE)
    hypo = np.dot(x,w)
    loss = hypo - y    
    cost = (np.sum(loss**2)+ld*np.sum(w**2)) / len(x)
    cost_a  = math.sqrt(cost)
    
    # stop when converge
    if old_cost == cost_a:
        break
    else:
        old_cost = cost_a  
    
    # calculate gradient
    gra = np.dot(x_t,loss) + ld*w
    
    # update paramteres by ADAM
    m = beta1*m+(1-beta1)*gra
    v = beta2*v+(1-beta2)*gra**2
    m_h = m/(1-beta1**(i+1))
    v_h = v/(1-beta2**(i+1))
    adam = m_h/(np.sqrt(v_h)+eps)
    w = w - l_rate * adam
    
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# save model
np.save('model.npy',w)
