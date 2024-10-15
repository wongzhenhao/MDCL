# RGCN code



## Running procedure

* cd to RGCN/
* run python file
```bash
python entity_classify_mdcl.py --dataset=dblp -e 150 --gpu=3
python entity_classify_mdcl.py --dataset=imdb -e 150 --gpu=3 --n-layers=3 --l2norm=1e-6 --n-hidden=32
python entity_classify_mdcl.py --dataset=acm -e 150 --gpu=3
```