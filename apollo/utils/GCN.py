import torch
import torchvision 
import torch.nn.functional as F
import torch_geometric.data as tgd
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from util import Util
from sklearn.manifold import TSNE
from IPython.display import Javascript  # Restrict height of output cell.
from sklearn.model_selection import ShuffleSplit
from GCN_Mult import GCN_Mult
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))


#model = GCN_Mult(hidden_channels=16,num_feats=train_vec[gvec_ind].num_features).double()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#criterion = torch.nn.CrossEntropyLoss()

class GCN(object):
    
    def __init__(self,train_vec,test_vec):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.train_vec = train_vec
        self.test_vec = test_vec
        self.model = GCN_Mult(hidden_channels=16,num_feats=train_vec.num_features).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        pass
        
    def visualize(self,h, color):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.show()
        
    def train(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(self.train_vec.x.double(), self.train_vec.edge_index)  # Perform a single forward pass.
        loss = self.criterion(out, self.train_vec.y.long())  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.model.eval()
        out = self.model(self.test_vec.x.double(), self.test_vec.edge_index).double()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred == self.test_vec.y.double()  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(self.test_vec.y)  # Derive ratio of correct predictions.
        return test_acc

