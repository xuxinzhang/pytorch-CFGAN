# -*- coding: utf-8 -*-
"""
Created on Sat May 23 14:38:23 2020

@author: zxx
"""

import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self,itemCount,info_shape):
        self.num_classes = 2
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount*2+info_shape,1024),


            nn.ReLU(True),
            nn.Linear(1024,128),
            nn.ReLU(True),
            nn.Linear(128,16),
            nn.ReLU(True),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,fakedata,realdata,condition):
        
        data_c = torch.cat((fakedata,realdata),1)
        data_c_label  = torch.cat((data_c,condition),1)
        result=self.dis( data_c_label )
        return result
    
  
    
class generator(nn.Module):

    def __init__(self,itemCount,info_shape):
        self.itemCount = itemCount
        self.latent_dim = 100
        self.num_classes = 2
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(self.itemCount+info_shape, 256),

            nn.ReLU(True),
            nn.Linear(256, 512),
#            nn.LeakyReLU(alpha=0.2)
            nn.ReLU(True),
            nn.Linear(512,1024),

            nn.ReLU(True),
            nn.Linear(1024, itemCount),
            nn.Tanh()
        )
    def forward(self,noise,useInfo):
        G_input = torch.cat([noise, useInfo], 1)
#        label_embedding = Flatten()(Embedding(self.itemCount, self.itemCount)(label))
        result=self.gen(G_input)
        return result   