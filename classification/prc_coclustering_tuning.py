# TTAを行わずに，単純にCLIPとViTのクラス分類性能を比較する
from copy import deepcopy
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sinkhorn_knopp import sinkhorn_knopp as skp
from sklearn.cluster import SpectralBiclustering
import clip

from domainbed import networks
from prc_functions.dataloader import get_clsdst_transform, get_dataloader
from prc_functions.vit_model import get_vit_model
from models.mine import Mine, MineTrainer
from loss.nt_xent import NTXentLoss


class NormalCLIP(nn.Module):
    def __init__(self, class_names):
        super().__init__()
        self.hparams = EasyDict({
            "clip_backbone": 'ViT-B/32',  # choice(['ViT-B/32', 'ViT-B/16', 'RN101']),
            "class_names": class_names,
            "num_domain_tokens": 16,
            'mlp_depth': 3,
            'mlp_width': 512,
            'mlp_dropout': 0.1,
            "loss": {
                "method": "mine",
                "nt_xent_temperature": 0.5,
            },
        })
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_DIM = 512
        # Normalizeしてからblock shuffleしている．block shuffleだけでは問題ないが，位相破壊も壊すとなると，順序は考えた方が良い．
        self.clsdst_transform = get_clsdst_transform()
        self.skp = skp.SinkhornKnopp()
        torch.autograd.set_detect_anomaly(True)
        self.mine_net = Mine().cuda()
        self.mine_trainer = MineTrainer(self.mine_net)
        self.mine_trainer.mine_net_optim = torch.optim.SGD(self.mine_net.parameters(), lr=0.001)
        if self.hparams["loss"]["method"] == "nt_xent":
            self.domain_criterion = NTXentLoss(self.device, 200, self.hparams["loss"]["nt_xent_temperature"])

        self._set_clip_models()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # Student model
        self.model = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * self.hparams['num_domain_tokens'], self.hparams).to(device=self.device, dtype=self.clip_model.dtype)
        self.model.apply(init_weights)


    def _set_clip_models(self):
        print(f"----- self.hparams['clip_backbone'] : {self.hparams['clip_backbone']}  -----")
        self.clip_model, preprocess = clip.load(self.hparams['clip_backbone'])
        self.clip_model = self.clip_model.cuda()
        # CLIPモデルのパラメータは更新させない
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        ##### Class Prompt用
        classnames = [f"a photo of a {name.replace('_', ' ')}" for name in self.hparams['class_names']]
        self.tokenized_prompts = torch.cat([clip.tokenize(c) for c in classnames]).cuda()
        self.text_features = self.clip_model.encode_text(self.tokenized_prompts)  # (クラス数, EMBEDDING_DIM)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        ##### domain Prompt用
        prompt_prefix = ' '.join(['X'] * self.hparams['num_domain_tokens'])
        domain_prompts = prompt_prefix + 'a photo of a '
        self.domain_tokenized_prompts = clip.tokenize(domain_prompts).to(self.device)  # (1, 77)
        with torch.no_grad():
            # トークン化されたテキストを入力し、トークンIDを埋め込みベクトルに変換する (トークン埋め込み).
            domain_embedding = self.clip_model.token_embedding(self.domain_tokenized_prompts).type(self.clip_model.dtype)  # (1, 77 (各文のトークン数（シーケンス長）), self.EMBEDDING_DIM)
        self.register_buffer('token_prefix', domain_embedding[:, :1, :])  # (1, 1, self.EMBEDDING_DIM)
        self.register_buffer('token_suffix', domain_embedding[:, self.hparams['num_domain_tokens'] + 1:, :])  # (1, 68, self.EMBEDDING_DIM)


    def _get_domain_text_features(self, image_features):
        domain_features = self.model(image_features)  # (B, EMBEDDING_DIM * num_domain_tokens)
        domain_features /= domain_features.norm(dim=-1, keepdim=True)
        domain_features = domain_features.reshape(domain_features.shape[0], self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)  # (B, EMBEDDING_DIM * num_domain_tokens)

        batch_size = image_features.shape[0]
        rep_token_prefix = self.token_prefix.repeat_interleave(batch_size, dim=0)  # (B, 1, EMBEDDING_DIM)
        rep_token_suffix = self.token_suffix.repeat_interleave(batch_size, dim=0)  # (B, 77-1-num_domain_tokens, EMBEDDING_DIM)
        domain_token_cat = torch.cat([rep_token_prefix, domain_features, rep_token_suffix], dim=1)  # (B, 77, EMBEDDING_DIM)
        
        # Transfromer
        x = domain_token_cat + self.clip_model.positional_embedding.type(self.clip_model.dtype)  # (B, 77, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (77, B, EMBEDDING_DIM)
        x = self.clip_model.transformer(x)  # (77, B, EMBEDDING_DIM)
        x = x.permute(1, 0, 2)  # (B, 77, EMBEDDING_DIM)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (B, 77, EMBEDDING_DIM)
        
        domain_text_features = x[torch.arange(x.shape[0]), self.domain_tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection  # (B, EMBEDDING_DIM))

        return domain_text_features


    def forward(self, x, x_clsdst):
        ##### Class
        image_features = self.clip_model.encode_image(x)  # (B, EMBEDDING_DIM)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()  # (B, Sourceクラス数)

        ##### Domain
        image_clsdst_features = self.clip_model.encode_image(x_clsdst)  # (B, EMBEDDING_DIM)
        domain_text_features = self._get_domain_text_features(image_clsdst_features)

        if self.hparams["loss"]["method"] == "nt_xent":
            ##### logits_per_imageに合わせねば.
            raise ValueError("returnを logits_per_imageに合わせねば.")
            loss = self.domain_criterion(image_clsdst_features, domain_text_features)
        elif self.hparams["loss"]["method"] == "mine":
            # 流石にデータセット単位で類似度計算を行うと，10,000*10,000の計算量となるので，バッチごとに行う．
            # そのため，バッチサイズは大きめでなくてはならない.
            sim_mtx = F.cosine_similarity(
                    x1=image_clsdst_features.unsqueeze(0),  # (B, B, EMBEDDING_DIM)
                    x2=domain_text_features.unsqueeze(1),  # (B, B, EMBEDDING_DIM)
                    dim=-1).detach().cpu().numpy()  # (B, B)
            bistochastic_mtx = self._get_bistochastic_mtx(sim_mtx)  # (B, B)
            clustered_mtx = self._biclustering(bistochastic_mtx)  # (B, B)

            diag = torch.diag(clustered_mtx).long()
            mean_feat_per_clust = [domain_text_features[diag == clust].mean(dim=0) for clust in range(torch.max(diag) + 1)]
            for i in range(len(mean_feat_per_clust)):
                for j in range(i+1, len(mean_feat_per_clust)):
                    data = torch.stack([mean_feat_per_clust[i], mean_feat_per_clust[j]], dim=1)
                    self.mine_trainer.train(data)
 
        return logits_per_image


    def _get_bistochastic_mtx(self, sim_mtx):
        """
            sim_mtx: Similarity Matrix (Square Matrix) ([-1, 1])
        """
        sim_mtx = (sim_mtx + 1) / 2  # [-1, 1]の類似度行列を[0, 1]に変換.
        bistochastic_mtx = self.skp.fit(sim_mtx)
        
        return bistochastic_mtx


    def _biclustering(self, scaled_sim_mtx, n_clusters=(4, 3)):
        """
            scaled_sim_mtx: 正規化した Similarity Matrix (Square Matrix) ([0, 1])
        """
        biclust_model = SpectralBiclustering(n_clusters=n_clusters, method="log", random_state=0)
        biclust_model.fit(scaled_sim_mtx)
        clustered_mtx = torch.tensor(np.outer(biclust_model.row_labels_, biclust_model.column_labels_))

        return clustered_mtx

        
    def evaluation(self, model, dataset, dataloader, model_name="clip"):
        correct = 0.
        len_dataloader = len(dataloader)
        print(f"dataloader: {len_dataloader}")
        for i, (imgs, labels) in tqdm(enumerate(dataloader)):
            imgs_clsdst = torch.stack([self.clsdst_transform(img) for img in imgs]).cuda()
            imgs, labels = imgs.cuda(), labels.cuda()
            output = model(imgs, imgs_clsdst)
            
            pred = output.argmax(dim=1)
            correct += (pred == labels).float().sum()

        accuracy = correct.item() / len(dataloader.dataset) * 100.
        print(f"Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    dataset_name = "cifar10"
    print(f"----- dataset: {dataset_name} -----")
    print(f"----- CLIP -----")
    clip_dataset, clip_loader = get_dataloader(dataset_name, does_resize = True)
    class_names = clip_dataset.classes
    clip_model = NormalCLIP(class_names)
    clip_model.eval()
    clip_model.evaluation(clip_model, clip_dataset, clip_loader, model_name="clip")

    print(f"----- ViT -----")
    vit_dataset, vit_loader   = get_dataloader(dataset_name, does_resize = False)
    vit_model = get_vit_model(dataset_name).cuda()
    vit_model.eval()
    vit_model.evaluation(vit_model, vit_dataset, vit_loader, model_name="vit")