import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob, pickle
from pathlib import Path 
import numpy as np
import tqdm, os
from mpl_toolkits.axes_grid1 import make_axes_locatable

BEH_LABELS = ['drink', 'eat', 'groom', 'hang', 'sniff', 'rear', 'rest', 'walk', 'eathand']

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.n_timesteps = 16

        self.X = pickle.load(open(data_path, 'rb'))
        self.preds, self.labels = [], []

        for series in tqdm.tqdm(self.X):
            for t in range(series.shape[0]-self.n_timesteps-1):
                self.preds.append(series[t:t+self.n_timesteps])
                self.labels.append(series[t+self.n_timesteps])

        self.preds = np.vstack(self.preds).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.n_datapoints = self.preds.shape[0]
 
    def __len__(self):
        'Denotes the total number of samples'
        return self.n_datapoints

    def __getitem__(self, index):

        'Generates one sample of data'
        return self.preds[index], self.labels[index]

class ARLSTM(torch.nn.Module):

   def __init__(self, embedding_dim, hidden_dim, output_dim):
      super(ARLSTM, self).__init__()
      self.hidden_dim = hidden_dim

      # The LSTM takes action labels as inputs, and outputs hidden states
      # with dimensionality hidden_dim.
      self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
      self.output_fn = torch.nn.Linear(hidden_dim, output_dim)
      self.act = F.log_softmax

   def forward(self, x):
      lstm_out, _ = self.lstm(x.transpose(-1,0).unsqueeze(-1))
      pred = self.output_fn(lstm_out[-1])
      
      # prepare for supervision only on the last step
      pred_scores = self.act(pred, dim=1)
      return pred_scores


def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

def compute_logprob(k, a, b):
    n = torch.zeros_like(a) + 9. # defining the support
    log_factorial_n = torch.lgamma(n + 1)
    log_factorial_k = torch.lgamma(k + 1)
    log_factorial_nmk = torch.lgamma(n - k + 1)

    return -1*(log_factorial_n - log_factorial_k - log_factorial_nmk +
                _log_beta(k + a, n - k + b) -
                _log_beta(b, a))

def train_fn(device):
    params = {'batch_size': 16384,
              'shuffle': False,
              'num_workers': 1}
    max_epochs = 300

    training_set = Dataset('/media/data_cifs/projects/prj_nih/prj_andrew_holmes/inference/batch_inference/scripts/HI_active_FC_preexposure_tokens.p')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    # set up the model
    model = ARLSTM(embedding_dim=1, hidden_dim=3, output_dim=9)    
    loss_fn = torch.nn.NLLLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    model.to(device)
    model.train()

    for epoch in range(max_epochs):

       # training loop
       for cnt, (batch, lab) in enumerate(training_generator):

          batch, lab = batch.to(device), lab.to(device)
          model.zero_grad()
          y = model(batch)
          loss = loss_fn(y, lab.long())
          loss.backward()
          optimizer.step()
          print('Epoch:{} | Batch: {}/{} | loss: {}'.format(epoch, cnt, training_generator.__len__(), loss.detach().item()))

    # saving model
    torch.save(model.state_dict(), 'models/FC_postcond_motif.pth')


def gen_sequence(model, seq, len_sample_traj):

    seq2 = seq.clone()
    pred_seq = list(seq.cpu().numpy().flatten())

    with torch.no_grad():
        for k in tqdm.tqdm(range(len_sample_traj)):
            pred = model(seq)
            
            # sample based on this distribution
            #next_label = torch.argmax(pred)
            #pred_seq.append(next_label.item())

            # SAMPLE based on softmax probabilities
            # TODO change to dirichlet
            probs = pred.exp()
            p = probs.squeeze().cpu().numpy()
            p = p/p.sum()
            next_label = np.random.choice(len(BEH_LABELS), p=p)
            pred_seq.append(next_label)
            
            # update seq
            seq2[:, :-1] = seq[:, 1:].data
            seq2[:, -1] = next_label
            seq = seq2.clone()

    return pred_seq

'''
Sample a bunch of synthetic action trajectories from the autoreg LSTM
'''
def simulate_seq(model, val_set, filename, r_idx=0, len_sample_traj=500, n_repeats=100):
    seq = val_set.preds[r_idx]
    #seq = torch.tensor([2, 2, 2, 5, 4, 4, 4, 4, 5, 3, 5, 5, 5, 2, 5, 2])
    #seq = seq.unsqueeze(0).float().to(device)
    seq = torch.tensor(seq).unsqueeze(0).float().to(device)

    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    sequences = np.zeros((n_repeats, len_sample_traj+16))    
    #sequences[0,:] = val_set.preds[val_set.rnd_idx[r_idx], val_set.rnd_start[r_idx]:val_set.rnd_start[r_idx]+16+len_sample_traj]


    for k in range(n_repeats):
        pred_seq = gen_sequence(model, seq, len_sample_traj)
        pred_seq = torch.tensor(pred_seq).unsqueeze(-1).unsqueeze(-1).to(device)
        zs, _ = model.lstm(pred_seq)
    
        sequences[k, :] = pred_seq.cpu().numpy().squeeze()

        #traj = [x.cpu().data.numpy().squeeze() for x in zs]
        #traj = np.vstack(traj)
        #ax.plot(traj[:,0], traj[:,1], traj[:,2])

    cmap = ax.imshow(sequences, interpolation='none', cmap='Set1')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cmap, cax=cax)

    cbar.set_ticks(np.arange(9))
    cbar.set_ticklabels(BEH_LABELS)
    plt.setp(cbar.ax.get_yticklabels(), fontsize=8)

    ax.set_xlabel('time')
    ax.set_xticks([0, len_sample_traj])
    ax.set_xticklabels(['0s', '%0.2fs'%(len_sample_traj/30)])
    ax.set_yticks([])
    #ax.set_yticklabels(['original'])
    #ax.set_ylabel('simulated', fontsize=8)
    #plt.axis('off')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    plt.savefig(os.path.join('simulated_seqs', filename))
    plt.close()


def eval_fn(device, save_path):
    np.random.seed(0)
    params = {'batch_size': 16,
              'shuffle': False,
              'num_workers': 1}

    val_set = Dataset('/media/data_cifs/projects/prj_nih/prj_andrew_holmes/inference/batch_inference/scripts/HI_active_FC_preexposure_tokens.p')
    val_generator = torch.utils.data.DataLoader(val_set, **params)

    model = ARLSTM(embedding_dim=1, hidden_dim=3, output_dim=9)    
    model.to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    '''
    #### VAL MODEL
    for cnt, (batch, lab) in enumerate(val_generator):

        batch, lab = batch.to(device), lab.to(device)
        y = model(batch)
        print(y, lab)
        import ipdb; ipdb.set_trace()

    import os; os._exit(0)
    '''

    for k in range(25):
        simulate_seq(model, val_set, 'HI_FC_preexp_%03d.png'%k, r_idx=k, len_sample_traj=250, n_repeats=50)

if __name__ == '__main__':
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    train_fn(device)
    #eval_fn(device, 'models/FC_preexp_motif.pth')
