# Code Reference: (https://github.com/dolphin-zs/Doc2EDAG)

import argparse
import os
import torch.distributed as dist

from dee.utils import set_basic_log_config, strtobool
from dee.dee_task import DEETask, DEETaskSetting
from dee.dee_helper import aggregate_task_eval_info, print_total_eval_info, print_single_vs_multi_performance

set_basic_log_config()


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_name', type=str, required=True,
                            help='Take Name')
    arg_parser.add_argument('--data_dir', type=str, default='./Data',
                            help='Data directory')
    arg_parser.add_argument('--exp_dir', type=str, default='./Exps',
                            help='Experiment directory')
    arg_parser.add_argument('--save_cpt_flag', type=strtobool, default=True,
                            help='Whether to save cpt for each epoch')
    arg_parser.add_argument('--skip_train', type=strtobool, default=False,
                            help='Whether to skip training')
    arg_parser.add_argument('--eval_model_names', type=str, default='GIT',
                            help="Models to be evaluated")
    arg_parser.add_argument('--re_eval_flag', type=strtobool, default=False,
                            help='Whether to re-evaluate previous predictions')

    # add task setting arguments
    for key, val in DEETaskSetting.base_attr_default_pairs:
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        else:
            arg_parser.add_argument('--'+key, type=type(val), default=val)

    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


if __name__ == '__main__':
    # in_argv = parse_args()
    in_argv = argparse.Namespace(add_greedy_dec=True,
                                 bert_model='bert-base-chinese',
                                 test_file_name='sample_test_1.json',
                                 train_file_name='sample_train_1.json',
                                 dev_file_name='sample_dev_1.json',
                                 # task_name='wzh_ccks_9+4_210219',
                                 task_name='CZ_330',
                                 train_batch_size=8,
                                 data_dir='./Data',
                                 dropout=0.25,
                                 eval_batch_size=1,
                                 eval_model_names='GIT',
                                 exp_dir='./Exps',
                                 ff_size=512,
                                 fp16=False,
                                 gradient_accumulation_steps=8,
                                 hidden_size=384,
                                 learning_rate=0.0001,
                                 local_rank=-1,
                                 loss_gamma=1.0,
                                 loss_lambda=0.05,
                                 max_sent_len=128,
                                 max_sent_num=64,
                                 min_teacher_prob=0.1,
                                 model_type='GIT',
                                 neg_field_loss_scaling=3.0,
                                 no_cuda=False,
                                 num_tf_layers=4,
                                 num_train_epochs=11,
                                 only_master_logging=True,
                                 optimize_on_cpu=False,
                                 re_eval_flag=False,
                                 rearrange_sent=False,
                                 resume_latest_cpt=True,
                                 save_cpt_flag=True,
                                 schedule_epoch_length=10,
                                 schedule_epoch_start=10,
                                 seed=99,
                                 seq_reduce_type='MaxPooling',
                                 skip_train=False,
                                 summary_dir_name='/tmp/Summary',
                                 cpt_file_name='GIT',
                                 use_bert=False,
                                 use_crf_layer=True,
                                 use_doc_enc=True,
                                 use_path_mem=True,
                                 use_scheduled_sampling=True,
                                 use_token_role=True,
                                 gcn_layer=3,  # prefer FNs over FPs
                                 ner_num_tf_layers=8,
                                 )
    task_dir = os.path.join(in_argv.exp_dir, in_argv.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    in_argv.model_dir = os.path.join(task_dir, "Model")
    in_argv.output_dir = os.path.join(task_dir, "Output")

    # in_argv must contain 'data_dir', 'model_dir', 'output_dir'
    dee_setting = DEETaskSetting(
        **in_argv.__dict__
    )
    dee_setting.summary_dir_name = os.path.join(task_dir, "Summary")

    # build task
    dee_task = DEETask(dee_setting, load_train=not in_argv.skip_train)

    if not in_argv.skip_train:
        # dump hyper-parameter settings
        if dee_task.is_master_node():
            fn = '{}.task_setting.json'.format(dee_setting.cpt_file_name)
            dee_setting.dump_to(task_dir, file_name=fn)

        dee_task.train(save_cpt_flag=in_argv.save_cpt_flag)
    else:
        dee_task.logging('Skip training')

    if dee_task.is_master_node():
        if in_argv.re_eval_flag:
            data_span_type2model_str2epoch_res_list = dee_task.reevaluate_dee_prediction(dump_flag=True)
        else:
            data_span_type2model_str2epoch_res_list = aggregate_task_eval_info(in_argv.output_dir, dump_flag=True)
        data_type = 'test'
        span_type = 'pred_span'
        metric_type = 'micro'
        mstr_bepoch_list = print_total_eval_info(
            data_span_type2model_str2epoch_res_list, metric_type=metric_type, span_type=span_type,
            model_str=in_argv.eval_model_names,
            target_set=data_type
        )
        print_single_vs_multi_performance(
            mstr_bepoch_list, in_argv.output_dir, dee_task.test_features,
            metric_type=metric_type, data_type=data_type, span_type=span_type
        )

    # ensure every processes exit at the same time
    if dist.is_initialized():
        dist.barrier()




