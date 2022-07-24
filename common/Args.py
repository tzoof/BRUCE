import argparse


def parse_args(input_arguments):
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument("--usePostUL", default=False, action='store_true')
    parser.add_argument("--useUserBert", default=False, action='store_true')
    parser.add_argument("--useUserBertV2", default=False, action='store_true')
    parser.add_argument("--bundleEmbeddings", default=False, action='store_true')
    parser.add_argument("--itemThenUser", default=False, action='store_true', help="concatenating <item,user> instead of the default <user,item>, not supported by all models")
    parser.add_argument("--embed_shape", type=int, default=12)
    parser.add_argument("--feed_forward_shape", type=int, default=16)
    parser.add_argument("--num_transformer_layers", type=int, default=3, help="number of transformer layers")
    parser.add_argument("--transformer_feedforward_dim", type=int, default=32)
    parser.add_argument("--dense_dropout", type=float, default=0.5)
    parser.add_argument("--transformer_dropout", type=float, default=0.2)
    parser.add_argument("--op_after_transformer", type=str, default="avg", help="from ['concat', 'bert', 'avg', 'sum']")
    parser.add_argument("--weights_initrange", type=float, default=0.1)

    # Data params
    parser.add_argument("--max_bundle_size", type=int, default=100)
    parser.add_argument("--use_graph_sampling", default=False, action='store_true')
    parser.add_argument("--overwrite_cache", default=False, action='store_true')
    parser.add_argument("--use_mini_test", default=False, action='store_true')

    # Training params
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--start_val_from", type=int, default=1000)
    parser.add_argument("--end_training_at", type=int, default=10000)
    parser.add_argument("--num_epochs", type=int, default=3900)
    parser.add_argument("--evaluate_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--dont_multi_task", default=False, action='store_true')
    parser.add_argument("--dont_test_last_epoch", default=False, action='store_true')

    # Continue previous training session
    parser.add_argument("--model_path", type=str, default="", help="Previous model to load for test / further train")
    parser.add_argument("--num_epochs_on_previous_train", type=int, default=6000)
    parser.add_argument("--load_random_state", default=True, type=bool, help="We should load the random states only of the data wasn't loaded from cache on the first train")

    # Test params
    parser.add_argument("--checkpoint_metric_index", type=int, default=5,
                        help="Which metric to choose the best checkpoint by, this is the index of the metric in the metrics list, 0 is recall@5, 5 is MAP@5")
    parser.add_argument("--test_like_BCGN", default=True)
    parser.add_argument("--measureLikeDam", default=False, action='store_true')
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--test_only", default=False)

    # Warmup
    parser.add_argument("--apply_warmup", default=False, action='store_true')
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Logging params
    parser.add_argument("--model_name", type=str, default="user-item-transformers")
    parser.add_argument("--bundle_selection_string", type=str, default="cropping")
    parser.add_argument("--embedding_string", type=str, default="user-item")
    parser.add_argument("--sampling_string", type=str, default="original")
    parser.add_argument("--dataset_string", type=str, default="Youshu", help="Youshu, Steam or NetEase")
    parser.add_argument("--description", type=str, default="debug", help="experiment description")
    parser.add_argument("--DontCreateDir", default=False,  action='store_true')

    # Pretraining params
    parser.add_argument("--pretrain_train_batch_size", type=int, default=1024)
    parser.add_argument("--pretrain_test_batch_size", type=int, default=1024)
    parser.add_argument("--run_pretrain", default=False,  action='store_true')
    parser.add_argument("--use_pretrained", default=False,  action='store_true')
    parser.add_argument("--pretrained_path", type=str, default="")
    parser.add_argument("--pretrained_bpr_path", type=str, default="")

    args = parser.parse_args()
    return args
