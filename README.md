# Research on a BERT Fine-tuning and LightGCN Fusion Recommendation Method for Middle School Math Problems
## Project Overview
This project explores a fusion recommendation method for middle school math problems, combining graph-based collaborative filtering and text-based semantic understanding.
The main workflow includes:
### 1.Data Collection or Construction:
Build or collect a dataset of student exercise records, containing information such as:  

-Which students attempted which problems  

-Whether the answers were correct or incorrect  

-The content (text) of each problem
### 2.Model Training:
- **LightGCN**

Construct a bipartite graph representing relationships between students, problems, and knowledge concepts  

Train a LightGCN model to capture collaborative patterns based on this graph  

- **BERT**    

Train or fine-tune a BERT model to extract semantic representations from problem texts
### 3.Fusion and Recommendation:
Integrate the embeddings produced by LightGCN and BERT  

Recommend suitable problems to each student based on the fused representations
## Goal
The aim is to leverage both interaction behaviors and problem semantics to achieve more accurate and personalized exercise recommendations for middle school students.

