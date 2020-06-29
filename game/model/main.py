from batchgen import BatchGenerator
from model import Net, CnnClassifier
import pickle
import numpy as np
import torch
from accuracy import Accuracy

def load_batches(type: str) -> BatchGenerator:
    data = pickle.load(open(f'../../data/{type}', 'rb'))
    return BatchGenerator(data, 100, False)

train_batches = load_batches('train')
test_batches = load_batches('test')

net = Net()
if torch.cuda.is_available():
    net = net.cuda()
lr = 0.001
wd = 0
clf = CnnClassifier(net, lr, wd)
acc = Accuracy()
for epoch in range(100):
    loss_arr = []
    acc.reset()
    for batch in train_batches:
        loss = clf.train(batch.data, batch.label)
        loss_arr.append(loss)
    
    for batch in test_batches:
        pred = clf.predict(batch.data)
        acc.update(pred, batch.label)
    
    mn_loss = np.mean(np.array(loss_arr))
    sdev = np.std(np.array(loss_arr))

    print(f'epoch {epoch+1}')
    print(f'\ttrain loss: {mn_loss:0.3f} +- {sdev:0.3f}')
    print(f'\tval acc: {str(acc)}')

torch.save(net.state_dict(), 'trained_model_2.pt')