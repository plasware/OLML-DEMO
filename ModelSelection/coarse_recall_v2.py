from ModelSelection.model_clustering import ModelClustering
import copy
import numpy as np
import math
import os

PATH = os.path.dirname(__file__)

class CoarseRecall:
    def __init__(self, task):
        self.task = task
        if task == "mnli":
            with open(PATH + '/leep_score_mnli.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
        elif task == "tweet":
            with open(PATH + '/leep_score_tweet.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
        elif task == "boolq":
            with open(PATH + '/leep_score_boolq.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
        elif task == "copa":
            with open(PATH + '/leep_score_copa.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
        elif task == "multirc":
            with open(PATH + '/leep_score_multirc.txt', 'r') as f:
                lines = f.readlines()
                self.leep = lines[1].split('\t')
        #print("Task: %s" % task)
        self.leep_exp = [math.exp(float(leep_score)) for leep_score in self.leep]
        self.model_clustering_instance = ModelClustering()
        self.model_clustering_result = self.model_clustering_instance.do_cluster()
        self.model_name = self.model_clustering_instance.models

    def recall(self):
        # step 1: None singleton recall
        # print("---------None Singleton Recall----------")
        model_score_avg = copy.deepcopy(self.model_clustering_instance.model_score_avg)
        model_cluster_none_singleton = []
        proxy_score_none_singleton = np.zeros(len(self.model_name)).tolist()
        cluster_representatives = []
        for cluster in self.model_clustering_result:
            # select representatives for each none singleton cluster
            if len(cluster) > 1:
                cluster_representative = -1
                cluster_max_score = 0
                for item in cluster:
                    model_cluster_none_singleton.append(item)
                    if cluster_representative == -1:
                        cluster_representative = item
                        cluster_max_score = model_score_avg[item]
                    else:
                        if model_score_avg[item] > cluster_max_score:
                            cluster_representative = item
                            cluster_max_score = model_score_avg[item]
                cluster_representatives.append(cluster_representative)
                """
                print("%d: %s selected as cluster representative, leep score: %s"
                      % (cluster_representative,
                         self.model_name[cluster_representative],
                         self.leep[cluster_representative]))
                """

                # calculate proxy score
                for item in cluster:
                    proxy_score_none_singleton[item] = model_score_avg[item] * self.leep_exp[item]

        #print("----result----")

        # step 2: Singleton recall
        #print("-----------Singleton Recall------------")
        for cluster in self.model_clustering_result:
            if len(cluster) == 1:
                curr_model_idx = cluster[0]
                curr_model_acc = np.array(self.model_clustering_instance.model_scores[curr_model_idx])
                curr_model_score = 0
                for item in cluster_representatives:
                    representative_acc = np.array(self.model_clustering_instance.model_scores[item])
                    cos_similarity = curr_model_acc.dot(representative_acc) / (
                            np.linalg.norm(curr_model_acc) * np.linalg.norm(representative_acc))
                    curr_model_score += (cos_similarity * self.leep_exp[item])
                curr_model_score /= len(cluster_representatives)
                proxy_score_none_singleton[curr_model_idx] = curr_model_score * model_score_avg[curr_model_idx]

        recall_result = []
        selected_cnt = 15
        while selected_cnt > 0:
            max_idx = proxy_score_none_singleton.index(max(proxy_score_none_singleton))
            recall_result.append(self.model_name[max_idx])
            print("%d: %s selected by proxy score: %s" % (
                max_idx, self.model_name[max_idx], str(proxy_score_none_singleton[max_idx])))
            selected_cnt -= 1
            proxy_score_none_singleton[max_idx] *= -1

        """
        result = '\n'.join(recall_result)
        with open(self.task + "_recall_result.txt", "w") as f:
            f.write(result)
        """

        return recall_result


if __name__ == "__main__":
    coarse_recall = CoarseRecall('mnli')
    recall_result = coarse_recall.recall()
    print(recall_result)
