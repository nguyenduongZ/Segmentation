# from torchmetrics.classification import Accuracy


# class Evaluator:
#     def __init__(self, args):
#         self.clf = args.clf_n_classes

#     def __call__(self, y_pred, y_true):
#         device = y_pred.device

#         task = 'multiclass' if self.clf != 1 else 'binary'
#         acc = Accuracy(task=task, num_classes=self.clf).to(device=device)(y_pred, y_true)
