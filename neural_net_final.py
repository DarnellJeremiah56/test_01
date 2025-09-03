import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def deriv_sigmoid2(x):
    return np.exp(-x)/((1 + np.exp(-x)) **2)

def mse_loss(y_pred, y_true):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        #从正态（高斯）分布中抽取随机样本
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 +self.b3)
        return o1

    def train(self, data, all_y_trues):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y - y_pred)

                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                self.w1 = self.w1 - learn_rate * (d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1)
                self.w2 = self.w2 - learn_rate * (d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2)
                self.b1 = self.b1 - learn_rate * (d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1)

                self.w3 = self.w3 - learn_rate * (d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3)
                self.w4 = self.w4 - learn_rate * (d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4)
                self.b2 = self.b2 - learn_rate * (d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2)

                self.w5 = self.w5 - learn_rate * (d_L_d_ypred * d_ypred_d_w5)
                self.w6 = self.w6 - learn_rate * (d_L_d_ypred * d_ypred_d_w6)
                self.b3 = self.b3 - learn_rate * (d_L_d_ypred * d_ypred_d_b3)

            if epoch % 10 == 0:
                #np.apply_along_axis将arr数组的每一个元素经过func函数变换形成的一个新数组
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print('epoch {} loss: {} y_preds: {}'.format(epoch, loss, y_preds))

data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
all_y_trues = np.array([1, 0, 0, 1])

network = OurNeuralNetwork()
network.train(data, all_y_trues)

emily = np.array([-7, -3])
frank = np.array([20, 2])
print('emily:{}'.format(network.feedforward(emily)))
print('frank:{}'.format(network.feedforward(frank)))




