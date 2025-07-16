# Meta-Learning cho Pseudo-Labels trong Community Detection

Dự án nghiên cứu về việc áp dụng Meta-Learning để cải thiện việc sinh và cập nhật pseudo-labels cho bài toán Community Detection trong các mạng phức tạp.

## Tổng quan

Pipeline này kết hợp Meta-Learning với Pseudo-Labels để cải thiện hiệu suất phát hiện cộng đồng trên các graph phức tạp. Ý tưởng chính là sử dụng meta-learner để học cách sinh và cập nhật pseudo-labels tối ưu cho từng loại graph/episode khác nhau.

## Kiến trúc tổng thể

### 1. Base Model
- Backbone: GCN/GAT/GIN (Graph Neural Networks)
- Hỗ trợ: Spectral clustering, modularity-based methods

### 2. Meta-Learner
- MAML (Model-Agnostic Meta-Learning)
- Reptile
- Custom meta-update strategies

### 3. Pseudo-label Module
- Node embedding similarity
- Clustering-based label generation
- Confidence-based label refinement

## Cấu trúc dự án

```
meta-pseudo-community/
├── data/                     # Dữ liệu và preprocessing
├── models/                   # Các mô hình
├── baselines/               # Các mô hình SOTA để so sánh
├── utils/                   # Utilities và helpers
├── experiments/             # Scripts thực nghiệm
├── evaluation/              # Đánh giá và metrics
├── configs/                 # Configuration files
├── results/                 # Kết quả thực nghiệm
└── notebooks/               # Jupyter notebooks cho phân tích

```

## Cài đặt

### Tự động (Khuyến nghị)
```bash
./setup.sh
```

### Thủ công
```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài đặt dependencies chính
pip install -r requirements.txt

# Cài đặt PyTorch Geometric (có thể cần cài thủ công)
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

**Lưu ý**: Dự án đã được tối ưu hóa cho Python 3.12. Một số dependencies nâng cao có thể cần cài đặt thủ công.

## Sử dụng

### 1. Chuẩn bị dữ liệu
```bash
python data/download_datasets.py
python data/preprocess.py
```

### 2. Huấn luyện mô hình
```bash
python experiments/train_meta_pseudo.py --config configs/meta_gcn.yaml
```

### 3. Đánh giá
```bash
python experiments/evaluate.py --model_path results/best_model.pth
```

### 4. So sánh với SOTA
```bash
python experiments/compare_baselines.py
```

## Bộ dữ liệu

### PyTorch Geometric datasets (Sẵn sàng):
- **Citation Networks**: Cora, CiteSeer, PubMed
- **Social Network**: Reddit  
- **E-commerce**: Amazon-Computers
- **Academic**: DBLP (co-author network)

### Synthetic datasets:
- LFR benchmark graphs với ground truth
- Stochastic Block Models

**Lưu ý**: Các dataset lớn từ SNAP (LiveJournal, Orkut, Youtube) có thể được download thủ công nếu cần.

## Baselines SOTA

- Traditional: DeepWalk, Node2Vec, Louvain, Leiden
- Deep Learning: GraphSAGE, GCN, GAT, DGI, DMoN
- Self-supervised: Deep Clustering, GCN-based pseudo-label

## Metrics đánh giá

- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)
- Modularity
- F1-score, Purity, Conductance

## Tác giả

[Tên tác giả]

## License

MIT License