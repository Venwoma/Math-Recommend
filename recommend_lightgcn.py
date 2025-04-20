import torch
import pickle
from torch_geometric.nn import LGConv
import torch.nn as nn

# 1. 定义 LightGCN 模型（必须跟训练时一样）
class LightGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(LightGCN, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        return LGConv()(x, edge_index)

# 2. 加载 node2id
with open('node2id.pkl', 'rb') as f:
    node2id = pickle.load(f)

# 3. 加载模型
model = LightGCN(num_nodes=len(node2id), embedding_dim=64)
model.load_state_dict(torch.load('lightgcn_model.pth'))
model.eval()

# 4. 加载测试集边
test_edge_index = torch.load('test_edge_index.pth')  # 你的训练时保存的测试集
# 注意：如果 test_edge_index.pth 没保存过，需要从训练脚本里加一句 torch.save(test_edge_index, 'test_edge_index.pth')

# 5. 得到节点的嵌入
with torch.no_grad():
    embeddings = model(test_edge_index)  # 注意，这里是用 测试集 构图的

# 6. 构造 ground truth（测试集中真实存在的边）
test_pos_edges = test_edge_index.t().tolist()  # [[user1, item1], [user2, item2], ...]

# 7. 推荐每个节点 top-K
K = 5  # 推荐Top-K个项目
num_nodes = embeddings.size(0)

# 计算所有节点之间的打分（内积）
score_matrix = torch.matmul(embeddings, embeddings.t())  # (num_nodes, num_nodes)

# 8. 对每个节点，排除掉自己，并取Top-K
_, topk_indices = torch.topk(score_matrix, K+1, dim=1)  # 每行取K+1个（包含自己）
topk_indices = topk_indices[:, 1:]  # 去掉自己（第一列）

# 9. 评估 Precision@K 和 Recall@K
hit = 0
total_pred = 0
total_true = 0

# 建立真实的边集合，方便快速查找
test_pos_set = set((u, v) for u, v in test_pos_edges)

for u in range(num_nodes):
    pred_items = topk_indices[u].tolist()  # u节点推荐的Top-K
    true_items = [v for (src, v) in test_pos_edges if src == u]  # u节点真实连的节点

    pred_set = set(pred_items)
    true_set = set(true_items)

    hit += len(pred_set & true_set)  # 预测正确的数量
    total_pred += len(pred_set)      # 总推荐的数量（K个）
    total_true += len(true_set)      # 真实喜欢的数量

precision = hit / total_pred if total_pred > 0 else 0
recall = hit / total_true if total_true > 0 else 0

print(f"测试集 Precision@{K}: {precision:.4f}")
print(f"测试集 Recall@{K}: {recall:.4f}")
