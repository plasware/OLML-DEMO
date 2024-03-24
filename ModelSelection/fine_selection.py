import pandas as pd
import json
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")
import os
import codecs
from sklearn.cluster import KMeans
import numpy as np

FILE_PATH = os.path.dirname(__file__)

# 获得每一个模型在所有数据集上的val和test表现
all_datasets = []
glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']
superglue_datasets = ['copa', 'boolq', 'wic', 'cb', 'multirc', ]
cls_datasets = ['imdb', 'amazon', 'yelp_review_full', 'yahoo_answers_topics', 'dbpedia_14']
new_datasets1 = ['wnli', 'xnli', 'anli', 'app_reviews', 'stsb']
new_datasets2 = ['trec', 'rotten_tomatoes', 'amazon_polarity', 'tweet_eval', 'sick', 'paws', 'financial_phrasebank']
new_datasets3 = ['mteb_stsbenchmark-sts', 'mteb-sts12-sts']  # 两个stsb数据集

all_datasets += glue_datasets
all_datasets += superglue_datasets
all_datasets += cls_datasets


class FineSelection:
    def __init__(self):
        self.df_dic = {}
        self.df_dic_val = {}
        self.all_model_name = []
        self.load()
        self.dataset_list = list(self.df_dic_val['bert-base-uncased'].keys())  # 假设所有模型都在相同的数据集上进行了训练

    def load(self):
        self.all_model_name = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'albert-base-v2']
        filename_list = os.listdir(FILE_PATH + '/fine_selection_files/instruction/new_model')
        for filename in filename_list:
            if filename == ".DS_Store":
                continue
            p1 = FILE_PATH + '/fine_selection_files/instruction/new_model/' + filename
            ori_tem = ''
            with codecs.open(p1, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f.readlines():
                    ori_tem += line
            sta = ori_tem.index('--model_name')
            end = ori_tem.index('--early_stopping')
            model_name = ori_tem[sta + 13:end - 1]
            self.all_model_name.append(model_name)

        for model_name in self.all_model_name:
            mn = model_name.replace('/', '--')
            self.df_dic[mn] = []

        for model_name in self.all_model_name:
            mn = model_name.replace('/', '--')
            self.df_dic_val[mn] = {}

        RESULT_PATH = FILE_PATH + '/fine_selection_files/results/'
        # p1 = '/Users/wenh/NLP/github/ProgressivePrompts/OLML_results/new_dataset_result-1/Capreolus--bert-base-msmarco/amazon_polarity.npy'
        # res = np.load(p1,allow_pickle=True).item()
        # print(res)
        for a in self.df_dic:
            count = 0
            for dataset in all_datasets:
                p1 = RESULT_PATH + a + '/' + dataset + '.npy'
                try:
                    res = np.load(p1, allow_pickle=True).item()
                    self.df_dic[a].append(round(res['test'], 2))
                    self.df_dic_val[a][dataset] = [round(ori_r, 2) for ori_r in res['val']]
                    count += 1
                except:
                    self.df_dic[a].append(0.0)
                    self.df_dic_val[a][dataset] = [0.0] * 5

        # new_datasets1是都没有的，要从path2找
        path2 = FILE_PATH + "/fine_selection_files/0512_result/"
        # new_dataset2是前四个模型没有的，要从path2找，否则从path1找
        path1 = FILE_PATH + '/fine_selection_files/new_dataset_result-1/'

        for a in self.df_dic:
            count = 0
            for dataset in new_datasets1:
                if dataset == 'stsb':
                    p1 = FILE_PATH + "/fine_selection_files/sts-b/" + \
                         a.replace('-', '_') + '/' + dataset + '_1000.npy'
                else:
                    p1 = path2 + a.replace('-', '_') + '/' + dataset + '_1000.npy'
                #         print(p1)
                try:
                    res = np.load(p1, allow_pickle=True).item()
                    self.df_dic[a].append(round(res['test'], 2))
                    self.df_dic_val[a][dataset] = [round(ori_r, 2) for ori_r in res['val']]
                    count += 1
                except:
                    self.df_dic[a].append(0.0)
                    self.df_dic_val[a][dataset] = [0.0] * 5

        for a in self.df_dic:
            count = 0
            for dataset in new_datasets2:
                if a in ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'albert-base-v2']:
                    p1 = path2 + a.replace('-', '_') + '/' + dataset + '_1000.npy'
                else:
                    p1 = path1 + a + '/' + dataset + '.npy'
                #         print(p1)

                try:
                    res = np.load(p1, allow_pickle=True).item()
                    self.df_dic[a].append(round(res['test'], 2))
                    self.df_dic_val[a][dataset] = [round(ori_r, 2) for ori_r in res['val']]
                    count += 1
                except:
                    self.df_dic[a].append(0.0)
                    self.df_dic_val[a][dataset] = [0.0] * 5

        path3 = FILE_PATH + '/fine_selection_files/528sts_2new/'
        # new_datasets3 5.28新增的两个stsb数据集
        for a in self.df_dic:
            count = 0
            for dataset in new_datasets3:
                p1 = path3 + dataset + '/' + a.replace('-', '_') + '/' + 'stsb_1000.npy'
                try:
                    res = np.load(p1, allow_pickle=True).item()
                    self.df_dic[a].append(round(res['test'], 2))
                    self.df_dic_val[a][dataset] = [round(ori_r, 2) for ori_r in res['val']]
                    count += 1
                except:
                    self.df_dic[a].append(0.0)
                    self.df_dic_val[a][dataset] = [0.0] * 5
        #             print(p1)

    # 该模型用于将目标模型在所有数据集（除了目标数据集）上的特定epoch的validation准确率进行聚类
    def create_cluster_model_new(self, target_model, target_dataset, target_epoch, K=4):
        # 提取训练数据
        x_train = []  # 存放所有数据集的第一个epoch的validation准确率
        y_train = []  # 存放所有数据集的测试准确率

        for i, dataset in enumerate(self.df_dic_val[target_model].keys()):
            if dataset == target_dataset:
                continue
            x_train.append(self.df_dic_val[target_model][dataset][target_epoch])
            y_train.append(self.df_dic[target_model][i])
        x_train = np.array(x_train).reshape(-1, 1)

        kmeans = KMeans(n_clusters=K)
        labels = kmeans.fit_predict(x_train)

        # 计算每个类别的平均测试准确率
        cluster_means = {}
        for i in range(4):
            cluster_means[i] = np.mean([y for y, label in zip(y_train, labels) if label == i])

        return kmeans, cluster_means

    # 模型在数据集上每次的validation结果r_v，模型在数据集上的test结果r_t 目标数据集d_t，目标模型m_t，总的训练步数S，validation间隔步数s_v，
    # 粗筛后的模型M_0=M,总的验证次数是T = S/s_v

    # 从t=0到T-1，遍历M_t每个模型，每个模型都在之前的基础上继续训练s_v步，如果M_t中模型数量>1：
    # 得到每一个模型对应的聚类模型，预测模型的最终表现，

    def filter_models(self, coarse_recall_result, num_models=10, threshold=0.1):
        results = {}  # 用于保存结果的字典
        target_dataset_list = ['tweet_eval', 'mnli', 'multirc', 'boolq']

        for dataset_index, target_dataset in enumerate(self.dataset_list):
            if target_dataset not in target_dataset_list:
                continue
            # 从df_dic中获取在当前数据集上表现最好的num_models个模型

            top_models = coarse_recall_result[:num_models]
            results[target_dataset] = {}
            for epoch in range(5):  # 假设你有5个epoch
                cur_num_models = len(top_models)
                # 针对每个模型，创建聚类模型并进行预测
                predictions = {}
                for target_model in top_models:
                    cluster_model, cluster_means = self.create_cluster_model_new(target_model, target_dataset, epoch, 3)
                    # 假设你的聚类模型可以接受一个模型的数据并返回一个预测值
                    val_accuracy = self.df_dic_val[target_model][target_dataset][epoch]
                    val_accuracy = np.array([val_accuracy]).reshape(-1, 1)

                    prediction = cluster_model.predict(val_accuracy)
                    predictions[target_model] = cluster_means[prediction[0]]
                # 从val表现最差的模型开始遍历，如果它的预测表现小于任何一个val表现比他好的模型所对应的预测表现，并且超过一定的阈值，就排除这个模型
                max_per = max(self.df_dic_val[x][target_dataset][epoch] for x in predictions)
                for i, target_model in enumerate(
                        sorted(predictions, key=lambda x: self.df_dic_val[x][target_dataset][epoch])):
                    if self.df_dic_val[target_model][target_dataset][epoch] == max_per:
                        continue
                    #   if predictions[target_model] < max(predictions[model] for model in top_models if df_dic_val[model][target_dataset][epoch] > df_dic_val[target_model][target_dataset][epoch])-threshold:
                    #       top_models.remove(target_model)
                    max_res = max(predictions[model] for model in top_models if
                                  self.df_dic_val[model][target_dataset][epoch] >
                                  self.df_dic_val[target_model][target_dataset][
                                      epoch])

                    if (max_res - predictions[target_model]) / max_res >= threshold:
                        top_models.remove(target_model)
                #                 # 如果筛选后的模型数量不到最初模型数量的一半，就移除在当前epoch的val表现最差的模型
                while len(top_models) > cur_num_models // 2:
                    worst_model = min(top_models, key=lambda model: self.df_dic_val[model][target_dataset][epoch])
                    top_models.remove(worst_model)
                # 保存结果
                results[target_dataset][epoch] = {
                    "left_models": list(top_models),
                    "left_num_models": len(top_models),
                    "best_test_performance": max(self.df_dic[model][dataset_index] for model in top_models)
                }

                # 只剩一个模型时，退出循环
                if len(top_models) == 1:
                    break
        return results


if __name__ == "__main__":
    fineSelection = FineSelection()
    threshold = 0.1
    coarse_recall_result = ['ishan--bert-base-uncased-mnli', 'Jeevesh8--feather_berts_46', 'emrecan--bert-base-multilingual-cased-snli_tr',
     'gchhablani--bert-base-cased-finetuned-rte', 'XSY--albert-base-v2-imdb-calssification', 'Jeevesh8--bert_ft_qqp-9',
     'Jeevesh8--bert_ft_qqp-40', 'connectivity--bert_ft_qqp-1', 'gchhablani--bert-base-cased-finetuned-wnli',
     'Jeevesh8--bert_ft_qqp-68', 'connectivity--bert_ft_qqp-96', 'classla--bcms-bertic-parlasent-bcs-ter',
     'Jeevesh8--init_bert_ft_qqp-33', 'connectivity--bert_ft_qqp-17', 'Jeevesh8--init_bert_ft_qqp-24']
    filter_results_0 = fineSelection.filter_models(coarse_recall_result, num_models=10, threshold=threshold)
    # print("Threshold: ", threshold)
    """
    for k in filter_results_0:
        print(k, filter_results_0[k])
        print('\n')
    """
