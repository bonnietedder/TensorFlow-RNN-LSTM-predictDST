#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=20         #隐层神经元的个数
lstm_layers=3       #隐层层数
input_size=7
output_size=1
lr=0.0006           #学习率

#——————————————————导入数据——————————————————————

f=open('/Users/apple/Desktop/tensorflow/DST/2002/2002dstdata.csv')
df=pd.read_csv(f)     #读入数据
data=df.iloc[:,2:10].values  #取第3-10列

#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=6120):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    #np.std axis=0计算每一列的标准差
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]
       y=normalized_train_data[i:i+time_step,7,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#获取测试集
def get_test_data(time_step=20,test_begin=6121):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y

#——————————————————定义神经网络变量——————————————————

#输入层、输出层权重、偏置、dropout参数
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')    

#——————————————————定义神经网络结构——————————————————

def lstmCell():
    #basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    #正则化
    #dropout：指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    #dropout只能是层与层之间（输入层与LSTM1层、LSTM1层与LSTM2层）的dropout
    #同一个层里面，T时刻与T+1时刻是不会dropout的
    return basicLstm

def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    #循环final_states至init_state
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #需要将tensor转成2维进行计算，weights是二维计算
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————训练模型————————————————————

def train_lstm(batch_size=60,time_step=20,train_begin=0,train_end=6120):
    # 训练样本中第1 - 6120个样本，每次取20个
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)

    print(np.array(train_x).shape)#6100 20 7 
    print(np.array(train_y).shape)#6100, 20, 1
    print(batch_index)
     #相当于总共6100句话，每句话20个字，每个字7个特征,对于这些样本每次训练60句话

    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
        #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    # the error between prediciton and real data  
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #class tf.train.AdamOptimizer
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            #每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]],keep_prob:0.5})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save/model.ckpt'))
        #I run the code on Linux, so use 'model_save/model.ckpt'
        #if you run it on Windows 10 , please use 'model_save\\model.ckpt'
        print("The train has finished")
train_lstm()

#————————————————预测模型————————————————————

def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    with tf.variable_scope("sec_lstm",reuse=tf.AUTO_REUSE):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]],keep_prob:1})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-(test_y[:len(test_predict)]))/(np.abs(test_predict+(test_y[:len(test_predict)]))))
         #偏差程度
        print("The accuracy of this predict:",acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',label='predict')
        plt.plot(list(range(len(test_y))), test_y,  color='r',label='actual')
        plt.title('2002')
        plt.ylabel('Dst Index /nT')
        plt.xlabel('month')
        plt.legend(loc='best')
        new_ticks=np.linspace(0,len(test_predict),12)
        plt.xticks(new_ticks,[r'$1$', r'$2$', r'$3$', r'$4$', r'$5$',r'$6$',r'$7$',r'$8$',r'$9$',r'$10$',r'$11$',r'$12$'])
        plt.show()

prediction()