from fastai.tabular import * 
from fastai.metrics import Precision, Recall, MatthewsCorreff, ConfusionMatrix
from sklearn.model_selection import train_test_split
import os

data = pd.read_csv('', encoding = 'utf8')
path = ''

counts = data['complexity_class'].value_counts()
rate = counts[1]/(counts[0] + counts[1])
rate

procs = [Categorify]
valid_idx = range(int(len(data)*0.9), len(data))

dep_var = ''
cat_names = ['eras']
emb_szs = {}
for name in cat_names:
    emb_szs[name] = 2
data = data.drop([],axis=1)

sample = data.sample(50000)
data = TabularDataBunch.from_df(os.getcwd(),data, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names) 

precision = Precision()
recall = Recall()
mattcor = MatthewsCorreff()
learn = tabular_learner(data, layers=[100,300,300,300,100], emb_szs=emb_szs, metrics=[accuracy,precision,recall,mattcor], ps=[0.5,0.5,0.5,0.5,0.5])
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(10, 1e-2)
learn.recorder.plot_metrics()

preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix()