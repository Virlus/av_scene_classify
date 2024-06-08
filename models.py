import torch
import torch.nn as nn


class MeanConcatDense(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_classes)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, self.num_classes)
        )
        # self.embedding = nn.Sequential(
        #     nn.Linear(audio_emb_dim+video_emb_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(512, 256),
        # )
        # self.outputlayer = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_classes),
        # )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        # combined_feat = torch.cat([audio_feat, video_feat], dim=-1)
        # combined_feat = combined_feat.mean(1)
        
        # embed = torch.cat((audio_emb, video_emb), 1)
        # embed = self.embedding(combined_feat)
        # output = self.outputlayer(embed)
        output = (audio_emb + video_emb) / 2.0
        return output

