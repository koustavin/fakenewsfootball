import sys
import pathlib
import logging
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score

# storing the original stdout, stderr values to global variables
original_stderr = sys.stderr.write
original_stdout = sys.stdout.write

def get_logfile(logfile_name = 'All.log'):
    '''
        the logfile name is non-mandatory argument, log path is predetermined
        using DEBUG as logging level
        returns the logger
    '''
    logfile = pathlib.Path('..') / 'Logs' / logfile_name
    
    LOGGER_NAME = 'kb'
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.DEBUG) # preset levels - NOTSET 0, DEBUG 10, INFO 20, WARNING 30, ERROR 40, CRITICAL 50

    log_handler = logging.FileHandler(logfile, mode = 'w+')
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(name)s %(message)s"))
    log.addHandler(log_handler)
    
    return log

def divert_stdOut2log(log):
    '''
        diverting the stdout to provided log file
        so that, whatever was supposed to print in the notebook would get written in the log file
        idea taken from https://stackoverflow.com/a/11325249/22459343, https://stackoverflow.com/a/50803365/22459343
    '''
    sys.stderr.write = log.error
    sys.stdout.write = log.info

def revert_log2stdOut():
    '''
        reverting stdout to original values
    '''
    sys.stderr.write = original_stderr
    sys.stdout.write = original_stdout

def test_results(y_test, y_pred, fit_time, pred_time):
    '''
        Print out various validation results
        takes actual and predicted labels of test set as input
        (and time taken to fit & predict, since the function is being imported now)
        Prints out the numbers, no return value
    '''
    precision, recall, fscore, train_support = score(y_test, y_pred,
                                                 labels = ['Real', 'Fake'], average='binary')
    print('Fit time: {} s / Predict time: {} s ---- Precision: {} / Recall: {} / Accuracy: {}'\
                                                  .format(round(fit_time, 3),
                                                          round(pred_time, 3),
                                                          round(precision, 3),
                                                          round(recall, 3),
                                                          round((y_pred == y_test).sum()/len(y_pred), 3)))
    
    print('\nClassification Report :\n')
    print(classification_report(y_test, y_pred, target_names = ['Real', 'Fake']))

    print('ROC AUC Score : {}\n'.format(roc_auc_score(y_test, y_pred)))
    
    print('\nConfusion Matrix :\n')
    ConfusionMatrixDisplay.from_predictions(y_test,
                                            y_pred,
                                            labels = [0, 1],
                                            display_labels = ['Real', 'Fake'],
                                            cmap = 'magma')