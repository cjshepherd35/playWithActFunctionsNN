import numpy as np
from tensorflow.keras.datasets import mnist
import convolutional as cn
import dense as dn
import reshape as re
# from keras.utils import np_utils
TF_ENABLE_ONEDNN_OPTS=0
import othernn as nn


x = np.random.randn(1,4,4)
# #mine
# conv = nn.Convolutional((1,4,4), 3, 2)
# reshape = nn.Reshape((2,2,2), (2*2*2,1))
# den = nn.Layer_Dense(2*2*2, 2)


# conv.forward(x)
# reshape.forward(conv.output)
# den.forward(reshape.output.T)
# print(reshape.output)
# print(den.output)




#theirs
conv = cn.Convolutional((1,4,4), 3, 2)
res = re.Reshape((2,2,2), (2*2*2,1))
den = dn.Dense(2*2*2,2)

con = conv.forward(x)
reshaped = res.forward(con)
print(reshaped)
out = den.forward(reshaped)
print(out)