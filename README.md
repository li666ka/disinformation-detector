# Diploma

## GNN Node Features

GIN та GraphSAGE моделі **не використовують** handcrafted features
(emotional, stylistic, social). Node features формуються з MiniLM
sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`):

- Article node: embedding від `article_text` (max 2000 chars)
- Tweet/Retweet/Reply nodes: embedding від `text` поля (max 500 chars)

Embedding розмірність: 384.

Inductive bias задається структурою графа поширення:

```
Article (root)
   ├── Tweet 1
   │   ├── Retweet 1.1
   │   ├── Reply 1.1
   │   │   └── Reply 1.1.1 (BFS до depth=N)
   │   └── ...
   └── Tweet 2
       └── ...
```

### 3.X.Y Особливості GNN-моделей

На відміну від класичних моделей (Naive Bayes) та трансформерів (DistilBERT),
де ми досліджували вплив додаткових ознак (emotional, stylistic, social),
GNN-моделі (GIN, GraphSAGE) використовують виключно семантичні представлення
текстів через MiniLM (sentence-transformers/all-MiniLM-L6-v2) як ознаки
вузлів графа.

Це обумовлено тим, що:

1. **GNN-моделі мають вбудований індуктивний bias** через структуру графа
   поширення новини (article → tweets → retweets/replies), який сам по собі
   несе соціальний контекст;

2. **Handcrafted features (emotional/stylistic/social)** — це article-level
   ознаки, які не природно вписуються у вузли графа гетерогенної структури
   (де вузли — статті, твіти, репости, відповіді);

3. **Cascade-features (depth, breadth, lifetime)** вже відображаються через
   саму топологію графа під час message passing.

Тому для GIN та GraphSAGE проводилось одне дослідження конфігурації —
без feature ablation, з варіюванням лише гіперпараметрів архітектури
(hidden_dim, num_layers, dropout, pooling/aggregator).
