import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)
        if use_cosine_similarity:
            self.similarity_function = self._cosine_simililarity
        else:
            self.similarity_function = self._dot_simililarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def _get_pos_neg_masks(self, positive=True):
        """ 行列の要素(xis, xis), (xis, xjs), (xjs, xjs) が0,それ以外が1の行列をつくり,類似度行列のフィルタを作る."""
        pos_pairs = torch.from_numpy(np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size))
        tril = torch.tril(torch.ones(2 * self.batch_size, 2 * self.batch_size))

        if positive:
            positive_masks = pos_pairs.type(torch.bool).to(self.device)
            return positive_masks
        else:
            negative_masks = (1 - tril - pos_pairs).type(torch.bool).to(self.device)
            return negative_masks


    @staticmethod
    def _dot_simililarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        similarity_matrix = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)  # (縦, 横)で類似度行列作成
        return similarity_matrix


    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        cosine_similarity = nn.CosineSimilarity(dim=-1)
        similarity_matrix = cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return similarity_matrix


    def forward(self, zis, zjs):
        """ 最初にバッチ内の全ての組合せの類似度行列を作り,正例か負例かでフィルタ掛けてInfoNCEを計算する. """
        representations = torch.cat([zjs, zis], dim=0)  # 縦に結合
        similarity_matrix = self.similarity_function(representations, representations)  # 類似度行列の作成
        # 類似度行列からpositive/negativeの要素を取り出す.
        positives = similarity_matrix[self._get_pos_neg_masks(positive=True)].view(self.batch_size, -1)  # 1次元化したものをviewで
        negatives = similarity_matrix[self._get_pos_neg_masks(positive=False)].view(self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)  # 横に結合.
        logits /= self.temperature
        labels = torch.zeros(self.batch_size).to(self.device).long()  # 1次元
        loss = self.criterion(logits, labels)  # logitsは重みでもある.重みが大きい(類似度が大きい)　-> lossは小さくなる

        return loss / self.batch_size
