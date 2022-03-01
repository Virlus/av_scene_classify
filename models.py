from turtle import forward
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
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        
        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputlayer(embed)
        return output


class l3_dense(nn.Module):

    def __init__(self,emb_dim,num_classes):

        super(l3_dense, self).__init__()
    
        self.num_classes = num_classes
        self.emb_dim = emb_dim        

        #self.layer_1 = nn.Linear(self.emb_dim, self.num_classes)   
        self.model = nn.Sequential(
          nn.Linear(self.emb_dim, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(512,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=0.2),

          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(64,self.num_classes)
        )
    
    def forward(self, x):
        y = self.model(x)
        return y

class l3_combine(nn.Module):

    def __init__(self,emb_dim,num_classes):
        super(l3_combine, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
            nn.Linear(self.emb_dim,128),
            nn.Linear(128,self.num_classes),
        )
    def forward(self, x):
        y = self.model(x)
        return y
