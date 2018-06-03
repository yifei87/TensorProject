import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# �� Variable ��������Щ���� tensor
x, y = torch.autograd.Variable(x), Variable(y)

# ��ͼ
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

import torch
import torch.nn.functional as F     # ��������������

class Net(torch.nn.Module):  # �̳� torch �� Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # �̳� __init__ ����
        # ����ÿ����ʲô������ʽ
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # ���ز��������
        self.predict = torch.nn.Linear(n_hidden, n_output)   # ������������

    def forward(self, x):   # ��ͬʱҲ�� Module �е� forward ����
        # ���򴫲�����ֵ, ��������������ֵ
        x = F.relu(self.hidden(x))      # ��������(���ز������ֵ)
        x = self.predict(x)             # ���ֵ
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net �Ľṹ

# optimizer ��ѵ���Ĺ���
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # ���� net �����в���, ѧϰ��
loss_func = torch.nn.MSELoss()      # Ԥ��ֵ����ʵֵ�������㹫ʽ (������)

plt.ion()   # ��ͼ
plt.show()
for t in range(100):
    prediction = net(x)     # ι�� net ѵ������ x, ���Ԥ��ֵ

    loss = loss_func(prediction, y)     # �������ߵ����

    optimizer.zero_grad()   # �����һ���Ĳ�����²���ֵ
    loss.backward()         # ���򴫲�, �����������ֵ
    optimizer.step()        # ����������ֵʩ�ӵ� net �� parameters ��


    # ����������
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
