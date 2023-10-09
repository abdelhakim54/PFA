import matplotlib.pyplot as p
p.switch_backend('agg')
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
from  tensorflow.keras.utils import to_categorical
import itertools


def SaveHistory(history,FileName):

# convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

# save to json:  
    # hist_json_file ='History/'+ FileName+'.json' 
    # with open(hist_json_file, mode='w') as f:
     # hist_df.to_json((f))

# or save to csv: 
    hist_csv_file ='History/'+ FileName+'.csv'
    with open(hist_csv_file, mode='w') as f:
     hist_df.to_csv(f)
#############################################################################     
     
def PlotFigures(history,MyModel,Magnification):
   H=list(history.history.keys())
   
   acc = history.history[H[1]]
   val_acc = history.history[H[3]]

# Retrieve a list of list results on training and test data
# sets for each training epoch
   loss = history.history[H[0]]
   val_loss = history.history[H[2]]

# Get number of epochs
   epochs = range(len(acc))

# Plot training and validation accuracy per epoch
   p.ioff()
   fig1 = p.figure()
   p.plot(epochs, acc,label='Train')
   p.plot(epochs, val_acc,label='Test')
   p.title('Training and testing accuracy')
   p.xlabel("Epoch #")
   p.ylabel("Accuracy")
   p.legend()
#   fig1 = p.gcf()
#   
#   p.draw()
   fig1.savefig('Model/'+Magnification+'_Training_Testing_Accuracy.png',dpi=300)
   p.close(fig1)
   print("[INFO] TRAINING & TESTING ACCURACY FIGURE SAVED...")
    #p.figure()

    # Plot training and validation loss per epoch
   fig2 = p.figure()
   p.plot(epochs, loss,label='Train')
   p.plot(epochs, val_loss,label='Test')
   p.title('Training and testing loss')
   p.xlabel("Epoch #")
   p.ylabel("Loss")
   p.legend()
#   fig2 = p.gcf()
#   
#   p.draw()
   fig2.savefig('Model/'+Magnification+'_Training_Testing_Loss.png',dpi=300)
   p.close(fig2)
   print("[INFO] TRAINING & TESTING LOSS FIGURE SAVED...")

  

   
def ConfutionMatrix_ClassificationReport(MyModel,Magnification,test_set,num_of_test_samples):
      
   print("\n\tConfusion Matrix and Classification Report")
   probabilities = MyModel.predict_generator(generator=test_set)
   y_true = test_set.classes
   y_pred = probabilities > 0.5



   conf_mat = confusion_matrix(y_true, y_pred)
   print('Confusion matrix:\n', conf_mat)
   TP = conf_mat[0][0]
   FP = conf_mat[0][1]
   TN = conf_mat[1][1]
   FN = conf_mat[1][0]
   #*******************************************************************************
   accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
   print('True positive = ', TP)
   print('False positive = ', FP)
   print('False negative = ', FN)
   print('True negative = ', TN)
   try:
    Sensitivity=TP/(TP+FN)
    Specifity=TN/(TN+FP)
    Precision=TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1Score=2*(Recall * Precision) /(Recall + Precision)
    a=TN/(TN+FP)
    b=TP/(TP+FN)
    ImbalancedAccuracy=(a+b)/2
   except ZeroDivisionError as err:
    Sensitivity=0
    Specifity=0
    accuracy=0
    Precision=0
    F1Score=0
    Recall=0
   print('\nAccuracy= ',accuracy,' Sensitivity= ',Sensitivity,' Specificity= ',Specifity,' \nPrecision= ',Precision,' Recall= ',Recall,' F1 Score= ',F1Score)
   print('\nImbalancedAccuracy=',ImbalancedAccuracy)
#*******************************************************************************
   
   labels = list(test_set.class_indices.keys())
   fig = p.figure()
   ax = fig.add_subplot(111) 
   
   cax = ax.matshow(conf_mat, cmap=p.cm.Blues)
   fig.colorbar(cax)

   thresh = np.mean(conf_mat)
   for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
       p.text(j, i, "{:0.2f}".format(conf_mat[i, j]),
       horizontalalignment="center",
       color="white" if conf_mat[i, j] > thresh else "black")
        
   ax.set_xticklabels([''] + labels)
   ax.set_yticklabels([''] + labels)
   ax.set_yticklabels(ax.get_yticklabels(), rotation=90,va="center")
   
   Resultat='Accuracy={:0.2f}%'.format(accuracy*100)+'  Sensitivity={:0.2f}%'.format(Sensitivity*100)+'  Specificity={:0.2f}%'.format(Specifity*100)
   p.xlabel('Predicted\n'+Resultat)
   p.ylabel('Expected')
   p.savefig('Model/'+Magnification+'_CM.png',dpi=300)
   print("[INFO] CONFUSION MATRIX SAVED...")
   
   
   print("\n\tClassification Report")
   target_names = list(test_set.class_indices.keys())
   report_data=classification_report(test_set.classes, y_pred, target_names=target_names)
   print(report_data)  
   
   #*******************************************************************************
   
   
   
   num_classes = test_set.num_classes
   y_test = to_categorical(test_set.classes, num_classes = num_classes)
   y_pred = to_categorical(y_pred, num_classes = num_classes)
        
   roc_auc=dict()
   fpr=dict()
   tpr=dict()
        
   for i in range(num_classes) :
       fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])
            
        # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
   all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
   mean_tpr = np.zeros_like(all_fpr)
   for i in range(num_classes):
         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
   mean_tpr /= num_classes

   fpr["macro"] = all_fpr
   tpr["macro"] = mean_tpr
   roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
   fig = p.figure()
   p.plot(fpr["micro"], tpr["micro"],
               label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
               color='deeppink', linestyle=':', linewidth=4)

   p.plot(fpr["macro"], tpr["macro"],
               label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
               color='navy', linestyle=':', linewidth=4)

   colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
   for i, color in zip(range(num_classes), colors):
            p.plot(fpr[i], tpr[i], color=color,
                   label='ROC curve of {0} class  (area = {1:0.2f})'
                   ''.format(target_names[i], roc_auc[i]))

   p.plot([0, 1], [0, 1], 'k--')
   p.xlim([0.0, 1.0])
   p.ylim([0.0, 1.05])
   p.xlabel('False Positive Rate')
   p.ylabel('True Positive Rate')
   p.title(' ROC')
   p.legend(loc="lower right")
   fig.savefig('Model/'+Magnification+'_ROC.png',dpi=300)
   p.close(fig)   
   print("[INFO] ROC CURVES SAVED...")
   
   
   
   
   
   
   

   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
