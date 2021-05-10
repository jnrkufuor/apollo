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
#display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))


class GCN_Mult(torch.nn.Module):
    
    def __init__(self, hidden_channels, num_feats):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        super(GCN_Mult, self).__init__()
        torch.manual_seed(12345)
        num_labels=2
        self.conv1 = GCNConv(num_feats, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_labels)

    def forward(self, x, edge_index):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


#model = GCN_Mult(hidden_channels=16,num_feats=train_vec[gvec_ind].num_features).double()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#criterion = torch.nn.CrossEntropyLoss()

class GCN(object):

    def __init__(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
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
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(train_vec[gvec_ind].x.double(), train_vec[gvec_ind].edge_index)  # Perform a single forward pass.
        loss = criterion(out, train_vec[gvec_ind].y.long())  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test(self):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        model.eval()
        out = model(test_vec[gvec_ind].x.double(), test_vec[gvec_ind].edge_index).double()
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred == test_vec[gvec_ind].y.double()  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / len(test_vec[gvec_ind].y)  # Derive ratio of correct predictions.
        return test_acc

