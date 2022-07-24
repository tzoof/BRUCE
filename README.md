# BRUCE - Bundle Recommendation Using Contextualized item Embeddings

This is our official implementation for the paper: BRUCE - Bundle Recommendation Using Contextualized item Embeddings<br/>

# Run the Code 
## Best Configurations
The BRUCE architecture is modular and combines several independent 
components which can be configured to best match the
data and task <br/>

Here are the best configurations for running BRUCE on each dataset.
Steam:<br/>
1. Train BPR model: 
```
MFbaseline/trainMF.py --train_only --size=12 --dataset_string=Steam --avg_items<br>
```
2. Train model usin pretrained BPR embeddings:
```
Main.py --dataset_string=Steam --description=bestConfig --op_after_transformer=avg --num_epochs=10000 --num_transformer_layers=1 --start_val_from=8000 --pretrained_bpr_path=<bprModelPath>.pkl --use_pretrained --dont_multi_task
```
Youshu:<br/>
1. Train BPR model: 
```
MFbaseline/trainMF.py --train_only --size=24 --dataset_string=Youshu --avg_items --dont_multi_task --op_after_transformer=bert
```
2. Train model usin pretrained BPR embeddings:
```
Main.py --dataset_string=Youshu --description=BestConfig --embed_shape=24 --weight_decay=0.0075 --num_epochs=7000 --start_val_from=4000 --useUserBertV2 --pretrained_bpr_path=<BprModelPath.pkl> --use_pretrained
```
NetEase:<br/>
1. Train BPR model: 
```
MFbaseline/trainMF.py --train_only --size=24 --dataset_string=NetEase --avg_items
```
2. Train model usin pretrained BPR embeddings:
```
Main.py  --dataset_string=NetEase --description=BestConfig --seed=111 --embed_shape=24 --weight_decay=0.0075 --useUserBert --num_epochs=7000 --batch_size=2048 --start_val_from=4000 --evaluate_every=500 --use_pretrained --pretrained_bpr_path=<bprModelPath>.pkl
```

### BRUCE Configurations
BRUCE code is modular and can be used and changed according to the need and task.
#### 1. Using BPR pretrained embeddings
The default configuration is to randomly initialize item embeddings. <br>
In order to use pretrained embeddings you need to do the following steps. <br>
a. Train a BPR model - 
MFbaseline/trainMF.py --train_only --size=<12-48> --dataset_string=<Youshu/NetE
/Steam> --avg_items<br>
The saved BPR model path should look like TrainedModels/bpr_user_avg_items_<datetime>.pkl")
b. Train with pretrained BPR embeddings: by adding the parameters --use_pretrained --pretrained_bpr_path=<modelPath.pkl>

#### 2. Integrating User Information
The following user integration techniques are supported: <br>
a. Concatenation of the user to each item. default option, you also need to pass the op_after_transformer elaborated in the next section. <br>
The models' code is under the PreUL dir. <br>
b. User first. by passing --useUserBert or --useUserBertV2 (the first shares the Transformer layer with the auxiliary task of items recommendation while the second does not). <br>
The models' code is under the UserBert dir. <br>
c. Post Transformer Integration. by passing --usePostUL,  you also need to pass the op_after_transformer elaborated in the next section. <br>
The models' code is under the PostUL dir. <br>

#### 3. Aggregation Methods
The aggregation method preformed after the Transformer layer, the following are supported: <br>
a. Concatenation. --op_after_transformer=concat<br>
b. Summation --op_after_transformer=sum<br>
c. Averaging --op_after_transformer=avg<br>
d. First item (BERT-like) aggregation --op_after_transformer=bert<br>
e. Bundle embedding BERT-like aggregation --bundleEmbeddings --op_after_transformer=bert


#### 4. Multi-task Learning
You can avoid the multi-task learning process by using the --dont_multi_task flag.


# Citation
If you use this code, please cite our paper. Thanks!
```
@inproceedings{
  author    = {Tzoof Avny Brosh and
               Amit Livne and
               Oren Sar Shalom and
               Bracha Shapira and
               Mark Last},
  title     = {BRUCE - Bundle Recommendation Using Contextualized item Embeddings},
  year      = {2022}
}

```

### Acknowledgements
Portions of this code are based on the [BGCN paper's code](https://github.com/cjx0525/BGCN) and the [DAM paper's code](https://github.com/yliuSYSU/DAM).`
