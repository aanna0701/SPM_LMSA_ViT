
class ViT_pooling(nn.Module):
    def __init__(self, in_height, in_width, num_nodes, inter_dimension, num_blocks, mlp_ratio=4, heads=8, num_classes=10, GA=False, dropout=True):
        super(ViT_pooling, self).__init__()

        self.inter_dimension = inter_dimension
        self.heads = heads

        self.in_size = (num_nodes + 1, inter_dimension)

        self.patch_embedding = Patch_Embedding(
            patch_size=int(math.sqrt((in_height * in_width) // num_nodes)), in_channels=3, inter_channels=inter_dimension)
        
        self.dropout = dropout
        
        self.classifier = Classifier_1d(
        num_classes=num_classes, in_channels=inter_dimension)
        self.positional_embedding = Positional_Embedding(
        spatial_dimension=num_nodes, inter_channels=inter_dimension)
        
        self.dropout_layer = nn.Dropout(0.1)
        
    
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.inter_dimension))
        
        self.transformers = nn.ModuleList()


    # def make_layer(self, num_blocks, block, mlp_ratio, GA_flag):
    #     layer_list = nn.ModuleList()
    #     for i in range(num_blocks):
    #         layer_list.append(
    #             block(self.in_size, self.inter_dimension, self.heads, mlp_ratio, GA_flag))

    #     return layer_list

        j = 0
        for i in range(len(num_blocks)):
            if not j+1 == len(num_blocks):
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA, True)
                j += 1
                self.in_channels = 2 * self.in_channels
                self.heads = 2 * self.heads
            else:
                self.make_layer(num_blocks[i], Transformer_Block, mlp_ratio, GA)
        
        self.classifier = Classifier_1d(
            num_classes=num_classes, in_channels=self.in_channels)        


    def make_layer(self, num_blocks, tr_block, mlp_ratio, GA_flag, pool_block=False):
        for i in range(num_blocks):
            self.transformers.append(
                tr_block(self.in_channels, self.inter_dimension, self.heads, mlp_ratio, GA_flag))
        if pool_block:
            self.transformers.append(
                Pooling_layer(self.in_channels))




    def forward(self, x):
        '''
        [shape]
            x : (B, 3, H, W)
            x_patch_embedded = (B, HW, C)
            x_tmp = (B, HW, C)
            x_out = (B, classes)
        '''
        x_patch_embedded = self.patch_embedding(x)
        x_tmp = self.positional_embedding(x_patch_embedded)
        cls_token = self.cls_token.expand(x_patch_embedded.size(0), self.cls_token.size(1), self.cls_token.size(2))

        
        
        if self.dropout:
            x_tmp = self.dropout_layer(x_tmp)
            cls_token = self.dropout_layer(cls_token)
        
        for i in range(len(self.transformers)):
            x_tmp, cls_token = self.transformers[i](x_tmp, cls_token, self.dropout)
                
        x_out = self.classifier(cls_token)

        return x_out
        
        
class GA_block(nn.Module):
    '''
    Class-token Embedding
    '''
    def __init__(self, in_size, in_channels):
        super(GA_block, self).__init__()
        # self.mlp = MLP_GA(in_channels, in_channels, 4)
        self.in_dimension = in_channels
        self.avgpool = nn.AvgPool1d(in_size[0]-2)
        # self.maxpool = nn.MaxPool1d(in_size[0]-2)
        # self.avgpool_2 = nn.AvgPool1d(in_size[0]-1)
        # self.maxpool_2 = nn.MaxPool1d(in_size[0]-1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.Linear = nn.Linear(in_channels, in_channels)
        
    def forward(self, x, cls_token, edge_aggregation):
        '''
            [shape]
            x : (B, HW+1, C)
            visual_token : (B, HW, C)
            cls_token_in : (B, 1, C)
            weight : (B, 1, HW)
            weight_softmax : (B, 1, HW)
            cls_token_out : (B, 1, C)
            out : (B, HW+1, C)     
        '''
        
        
        
        edge_aggregation_avgpool = self.avgpool(edge_aggregation)
        node_importance = self.softmax(edge_aggregation_avgpool)

        
        node_aggregation = torch.matmul(node_importance.permute(0, 2, 1), x[:, 1:])
        total_aggregation = self.sigmoid(node_aggregation)
        
        total_aggregation_out = torch.mul(total_aggregation, self.Linear(cls_token))
        

        
        cls_token_out = cls_token + total_aggregation_out
        
        # print('node_importance: {}'.format(node_importance[0].norm(2)))
        # print('node_aggregation: {}'.format(node_aggregation[0].norm(2)))
        # print('total_aggregation: {}'.format(total_aggregation[0].norm(2)))
        # print('total_aggregation_out: {}'.format(total_aggregation_out[0].norm(2)))
        # print('cls_token: {}'.format(cls_token[0].norm(2)))
        # print('cls_token_out: {}'.format(cls_token_out[0].norm(2)))
        # print('==========================================')
        
        out = torch.cat((cls_token_out, x[:, 1:]), dim=1)
        
        
        return out
    
    
            edge_aggregation = self.Linear(edge.permute(0, 3, 2, 1))
            edge_aggregation = edge_aggregation.squeeze(-1)[:, 1:, 1:]