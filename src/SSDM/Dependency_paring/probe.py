"""Classes for specifying probe pytorch modules."""

import torch.nn as nn
import torch


class Probe(nn.Module):
    pass


class TwoWordPSDProbe(Probe):
    """ Computes squared L2 distance after projection by a matrix.

  For a batch of sentences, computes all n^2 pairs of distances
  for each sentence in the batch.
  """

    def __init__(self, use_cuda=True, model_dim=768):
        print('Constructing TwoWordPSDProbe')
        super(TwoWordPSDProbe, self).__init__()
        self.probe_rank = 32
        self.model_dim = model_dim
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        if use_cuda:
            self.to('cuda')
        else:
            self.to('cpu')

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
    """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class OneWordPSDProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, use_cuda=True, model_dim=768):
        print('Constructing OneWordPSDProbe')
        super(OneWordPSDProbe, self).__init__()
        self.probe_rank = 128
        self.model_dim = model_dim
        self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        if use_cuda:
            self.to('cuda')
        else:
            self.to('cpu')

    def forward(self, batch):
        """ Computes all n depths after projection
    for each sentence in a batch.

    Computes (Bh_i)^T(Bh_i) for all i

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(transformed.view(batchlen * seqlen, 1, rank),
                          transformed.view(batchlen * seqlen, rank, 1))
        norms = norms.view(batchlen, seqlen)
        return norms
