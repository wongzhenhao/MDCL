# GCN and GAT Code

## Running Procedure
```
python run_mdcl.py --dataset DBLP --model-type gat
python run_mdcl.py --dataset DBLP --model-type gcn --weight-decay 1e-6 --lr 1e-3

python run_mdcl.py --dataset ACM --model-type gat --feats-type 2
python run_mdcl.py --dataset ACM --model-type gcn --weight-decay 1e-6 --lr 1e-3 --feats-type=0

python run_multi_mdcl.py --dataset IMDB --model-type gat --feats-type 0 --num-layers 4
python run_multi_mdcl.py --dataset IMDB --model-type gcn --feats-type 0 --num-layers 3
```

