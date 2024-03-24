from ModelSelection.coarse_recall_v2 import CoarseRecall
from ModelSelection.fine_selection import FineSelection


class ModelSelection:
    def __init__(self, task='mnli'):
        self.task = task
        self.coarseRecall = CoarseRecall(task)
        self.fineSelection = FineSelection()

    def select(self):
        coarse_recall_result = self.coarseRecall.recall()
        fine_selection_result = self.fineSelection.filter_models \
            (coarse_recall_result=coarse_recall_result)[self.task]
        last_select = max(fine_selection_result.keys())

        return fine_selection_result[last_select]


if __name__ == "__main__":
    modelSelection = ModelSelection()
    select_result = modelSelection.select()
    print(select_result['left_models'][0])
