import tensorflow as tf
import argparse

from data.trec.batch_loader import BatchLoader as trec_loader
from data.msqc.batch_loader import BatchLoader as msqc_loader
from models.hier_qc import HierRNN


def parse_input():
    parser = argparse.ArgumentParser(description="Parse network configuretion")
    parser.add_argument("--data_type",type=str, default="trec")
    parser.add_argument("--epoch_num",type=int, default=300)
    parser.add_argument("--batch_size",type=int, default=100)
    parser.add_argument("--max_sent_len",type=int, default=20)
    parser.add_argument("--max_word_len",type=int, default=16)
    parser.add_argument("--char_embed_dim",type=int, default=20)
    parser.add_argument("--first_unit_size",type=int, default=40)
    parser.add_argument("--secod_unit_size",type=int, default=40)
    parser.add_argument("--highway_layers",type=int, default=1)
    parser.add_argument("--learning_rate",type=float, default=1.0e-4)
    parser.add_argument("--learning_rate_decay",type=int, default=1)
    parser.add_argument("--reg_lambda",type=float, default=0.1)
    parser.add_argument("--clip_norm",type=float, default=10.0)
    parser.add_argument("--cell_name",type=str, default="arnn")
    parser.add_argument("--K",type=int, default=2)

    parser.add_argument("--is_test",type=bool, default=False)
    parser.add_argument("--ckp_dir",type=str, default='checkpoint')
    parser.add_argument("--ckp_name",type=str, default='100.pkl')
    parser.add_argument("--result_dir",type=str, default="result")
    parser.add_argument("--test_per_batch",type=int, default=-1)

    return parser.parse_args()

if __name__ == "__main__":
    config = parse_input()
    with tf.Session() as sess:
        if config.data_type == "msqc":
            loader = msqc_loader(config)
        else:
            loader = trec_loader(config)

        model = HierRNN(sess=sess, loader=loader, config=config)
        model.run(config.epoch_num)
