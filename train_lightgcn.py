import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
import pickle
import random

# 1. 加载数据
edge_index = torch.load('edge_index.pth')
with open('node2id.pkl', 'rb') as f:
    node2id = pickle.load(f)

# 2. 划分数据集：训练集、验证集和测试集
edges = edge_index.t().tolist()  # 将边转换为列表格式，方便划分

# 随机打乱边列表
random.shuffle(edges)

# 划分为训练集、验证集、测试集
train_edges = edges[:int(0.9 * len(edges))]  # 90% 作为训练集
val_edges = edges[int(0.9 * len(edges)):int(0.95 * len(edges))]  # 5% 作为验证集
test_edges = edges[int(0.95 * len(edges)):]  # 5% 作为测试集

# 将这些边分别转为 PyTorch tensor
train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
val_edge_index = torch.tensor(val_edges, dtype=torch.long).t()
test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()

# 打印划分结果
print(f"训练集边数：{train_edge_index.size(1)}")
print(f"验证集边数：{val_edge_index.size(1)}")
print(f"测试集边数：{test_edge_index.size(1)}")


# 3. 模拟模型输入（你可以根据需要替换成你实际的模型结构）
class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LightGCN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        return LGConv()(x, edge_index)


# 4. 创建 LightGCN 模型
model = LightGCN(num_nodes=len(node2id), embedding_dim=256)

# 5. 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 6. 定义链接预测损失函数
def link_prediction_loss(pos_edge_index, neg_edge_index, embeddings):
    pos_score = torch.sum(embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]], dim=-1)
    neg_score = torch.sum(embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]], dim=-1)
    loss = torch.mean(torch.sigmoid(neg_score) - torch.sigmoid(pos_score))  # 使用 Sigmoid 损失
    return loss

# 7. 创建负采样
# 负采样函数，避免采到已存在的边
def negative_sampling(edge_index, num_nodes):
    positive_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    neg_edge_index = []
    
    while len(neg_edge_index) < edge_index.size(1):  # 采样数目与正边一样
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()

        # 确保u和v的边不存在于正边集合中
        while (u, v) in positive_edges or (v, u) in positive_edges:
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()

        neg_edge_index.append((u, v))

    return torch.tensor(neg_edge_index).t()

# 8. 训练 LightGCN
num_epochs = 200
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()
    embeddings = model(train_edge_index)  # 使用训练集边进行训练
    
    # 获取正边和负边（基于训练集）
    neg_edge_index = negative_sampling(train_edge_index, len(node2id))

    # 计算损失
    loss = link_prediction_loss(train_edge_index, neg_edge_index, embeddings)
    
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 训练完成后保存模型
torch.save(model.state_dict(), 'lightgcn_model.pth')
torch.save(test_edge_index, 'test_edge_index.pth')

# 9. 验证（在训练完成后进行）
model.eval()

# 使用验证集评估
val_embeddings = model(val_edge_index)
val_neg_edge_index = negative_sampling(val_edge_index, len(node2id))
val_loss = link_prediction_loss(val_edge_index, val_neg_edge_index, val_embeddings)
print(f"验证集损失：{val_loss.item()}")
