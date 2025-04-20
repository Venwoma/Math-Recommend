import pandas as pd
import networkx as nx
import torch
import pickle

# 1. 读取CSV数据
df = pd.read_csv(r"E:\user\用户目录\skill_builder_data.csv", dtype={"skill_name": str}, low_memory=False)

# 2. 创建一个无向图（LightGCN 用的是无向图）
G = nx.Graph()

# 3. 添加user-problem边
for _, row in df.iterrows():
    user = f'u_{row["user_id"]}'
    problem = f'p_{row["problem_id"]}'
    G.add_edge(user, problem, type='user-problem')

# 4. 添加problem-skill边
for _, row in df.iterrows():
    problem = f'p_{row["problem_id"]}'
    skill = f's_{row["skill_id"]}'
    G.add_edge(problem, skill, type='problem-skill')

# 5. 简单查看图结构
print(f"节点总数：{G.number_of_nodes()}")
print(f"边总数：{G.number_of_edges()}")
print("示例边:", list(G.edges(data=True))[:10])

# 6. 将 NetworkX 图转换为 LightGCN 输入格式 (edge_index)
# 创建节点到索引的映射
node2id = {node: i for i, node in enumerate(G.nodes())}

# 获取所有的边（包括用户-题目边和题目-技能边）
edges = [(node2id[u], node2id[v]) for u, v in G.edges()]

# 转换为 PyTorch 格式
edge_index = torch.tensor(edges, dtype=torch.long).t()

# 显示转换后的 edge_index
print("edge_index：", edge_index)

# 保存 edge_index
torch.save(edge_index, 'edge_index.pth')

# 保存 node2id
with open('node2id.pkl', 'wb') as f:
    pickle.dump(node2id, f)

# 标记节点类型
node_types = {}
for node, idx in node2id.items():
    if node.startswith('u_'):
        node_types[idx] = 'user'
    elif node.startswith('p_'):
        node_types[idx] = 'problem'
    elif node.startswith('s_'):
        node_types[idx] = 'skill'

# 取出用户节点、题目节点、技能节点
user_nodes = [idx for idx, t in node_types.items() if t == 'user']
problem_nodes = [idx for idx, t in node_types.items() if t == 'problem']
skill_nodes = [idx for idx, t in node_types.items() if t == 'skill']

# 打印节点数
print(f"用户节点数：{len(user_nodes)}")
print(f"题目节点数：{len(problem_nodes)}")
print(f"技能节点数：{len(skill_nodes)}")
