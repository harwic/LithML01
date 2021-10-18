#   注意！本程式需"dataset_1_psm資料檔"，執行"pgm_07_A.sas"取得。
import os
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

def main():
    PATH = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    dataset_path = '%s\\SAS_output\\dataset_1_psm.sas7bdat'%PATH
    if not os.path.isabs(dataset_path):
        print('Error: %s no found.'%dataset_path)
        return
    dataset = pd.read_sas(dataset_path, encoding='iso-8859-1')
    case_name = '1-DM-1-All'
    model_path = '%s\\PY_output\\model\\%s'%(PATH, case_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        pass
    
    #   篩選資料
    drop_list = []
    for i in list(dataset.index):
        if dataset['GP'][i] > 1:
            drop_list.append(i)
        elif dataset['DIFFTM'][i] > 16 or dataset['DIFFTM'][i] < 8:
            drop_list.append(i)
    dataset = dataset.drop(index=drop_list)
    dataset.index = range(0, len(dataset.index))
    drop_list = []
    for i in list(dataset.columns):
        flag = False
        if i.find('DG1_') != -1 and i.find('DG1_DD') == -1:
            drop_list.append(i)
            cnt = 0
            for j in list(dataset.index):
                if dataset[i][j] == 0:
                    cnt += 1
            if cnt == len(dataset.index) or cnt == 0:
                drop_list.append(i[:i.index('_')]+'_DD'+i[i.index('_'):])
        elif i.find('LAB_') != -1:
            flag = False
        elif i.find('DG1_') != -1 and i.find('DG1_DD') != -1:
            flag = False
        elif i.find('DG2_') != -1:
            flag = False
            cnt0 = 0
            cnt1 = 0
            for j in list(dataset.index):
                if dataset[i][j] == 0:
                    cnt0 += 1
                else:
                    cnt1 += 1
            if cnt0 == len(dataset.index) or cnt1 == len(dataset.index):
                flag = True
        elif i.find('CCI_') != -1:
            flag = False
        if flag:
            drop_list.append(i)
    dataset = dataset.drop(columns=drop_list)
    
    #   設定Label
    Y = []
    for i in list(dataset.index):
        if dataset['LABVAL'][i] < 0.6:
            Y.append(0)
        else:
            Y.append(1)
    Y = pd.Series(Y)
    
    #   切資料集
    X = dataset.copy()
    #   --------------------------------------------------
    for i in list(X.columns):
        if i in ['AGE', 'DIFFTM', 'LASTDOSE', 'DAILYDOSE', 'LASTFREQ', 'HG', 'WG', 'SBP', 'DBP'] or i.find('LAB_') != -1 or i.find('DG1_DD') != -1:
            data = []
            for j in list(X.index):
                data.append(X[i][j])
            data = np.array(data).reshape(-1, 1)
            data = MinMaxScaler().fit(data).transform(data)
            tmp = []
            for j in range(0, len(data)):
                tmp.append(data[j][0])
            X.loc[:, i] = tmp
    #   --------------------------------------------------
    tmp_indx, test_indx, opd_indx = [], [], []
    for i in list(X.index):
        if X['GP'][i] == -1:
            opd_indx.append(i)
        elif X['FLAG'][i].find('NO') != -1:
            tmp_indx.append(i)
        else:
            test_indx.append(i)
    X = X.drop(columns=['LABVAL', 'No', 'GP', 'FLAG', '_PS_', '_MATCHWGT_'])
    tmp = test_indx + opd_indx
    tmp_X = X.drop(index=tmp)
    tmp_X.index = range(0, len(tmp_indx))
    tmp_Y = Y.drop(index=tmp)
    tmp_Y.index = range(0, len(tmp_indx))
    tmp = tmp_indx + opd_indx
    x_test = X.drop(index=tmp)
    x_test.index = range(0, len(test_indx))
    y_test = Y.drop(index=tmp)
    y_test.index = range(0, len(test_indx))
    tmp = tmp_indx + test_indx
    x_opd = X.drop(index=tmp)
    x_opd.index = range(0, len(opd_indx))
    y_opd = Y.drop(index=tmp)
    y_opd.index = range(0, len(opd_indx))
    '''
    param = {'n_estimators':list(range(100,201,10)), 'max_depth':list(range(3,11))}
    grid = GridSearchCV(xgb.XGBClassifier(learning_rate=0.01,
                                          use_label_encoder=False), param_grid=param, cv=5)
    grid.fit(tmp_X, tmp_Y)
    print(grid.best_params_)
    '''
    #   建模
    SKF = StratifiedKFold(n_splits=5)
    k = 1
    df1 = []
    df2 = [['Ans']+list(y_test)]
    df3 = [['Ans']+list(y_opd)]
    for train_indx, val_indx in SKF.split(tmp_X, tmp_Y):
        x_train = tmp_X.drop(index=val_indx)
        x_train.index = range(0, len(train_indx))
        y_train = tmp_Y.drop(index=val_indx)
        y_train.index = range(0, len(train_indx))
        x_val = tmp_X.drop(index=train_indx)
        x_val.index = range(0, len(val_indx))
        y_val = tmp_Y.drop(index=train_indx)
        y_val.index = range(0, len(val_indx))
        df1, df2, df3 = LR_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path)
        df1, df2, df3 = SVM_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path)
        df1, df2, df3 = RF_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path)
        df1, df2, df3 = XGB_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path)
        k += 1
    ft_list = ['Name', 'Model', 'Set', 'K', 'TP', 'FP', 'TN', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'ACC', 'F1', 'LR+', 'LR-', 'DOR', 'AUC', 'Brier']
    df1 = pd.DataFrame(df1, columns=ft_list)
    df2 = pd.DataFrame(df2, columns=list(range(0, len(df2[0]))))
    df3 = pd.DataFrame(df3, columns=list(range(0, len(df3[0]))))
    df1, df2 = ENSB_Predictor(df1, df2, 'ens_test')
    df1, df3 = ENSB_Predictor(df1, df3, 'ens_opd')
    df4 = KP_Calculator(df2)
    df5 = x_test.copy()
    df5.loc[:, 'Type'] = 'data'
    df5.loc[:, 'Model'] = ''
    df5.loc[:, 'K'] = ''
    df5.loc[:, 'N'] = ''
    tmp = ['Type', 'Model', 'K', 'N'] + list(x_test.columns)
    df5 = df5[tmp]
    df6, df7 = AAA(df2, df5)
    df6 = pd.DataFrame(df6, columns=list(df5.columns))
    df7 = pd.concat([df7, df6])
    file_path = '%s\\PY_output\\rslt-1\\rslt-%s.xlsx'%(PATH, case_name)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Score', index=False)
        df2.to_excel(writer, sheet_name='Pred_test', index=False)
        df3.to_excel(writer, sheet_name='Pred_opd', index=False)
        df4.to_excel(writer, sheet_name='Kappa_test', index=False)
        df7.to_excel(writer, sheet_name='Dataset_test', index=False)
    writer.close()
    return

def LR_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'LogR'
    model = LogisticRegression().fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    tmp = CM_Calculator(model_name, 'train', k, y_train, y_pred)
    y_prob = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_train, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_val)
    tmp = CM_Calculator(model_name, 'val', k, y_val, y_pred)
    y_prob = model.predict_proba(x_val)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_val, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_test)
    tmp = CM_Calculator(model_name, 'test', k, y_test, y_pred)
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_test, y_prob))
    df1.append(tmp)
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    tmp = CM_Calculator(model_name, 'opd', k, y_opd, y_pred)
    y_prob = model.predict_proba(x_opd)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_opd, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_opd, y_prob))
    df1.append(tmp)
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def SVM_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'SVC'
    model = SVC(C=1,
                probability=True).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    tmp = CM_Calculator(model_name, 'train', k, y_train, y_pred)
    y_prob = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_train, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_val)
    tmp = CM_Calculator(model_name, 'val', k, y_val, y_pred)
    y_prob = model.predict_proba(x_val)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_val, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_test)
    tmp = CM_Calculator(model_name, 'test', k, y_test, y_pred)
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_test, y_prob))
    df1.append(tmp)
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    tmp = CM_Calculator(model_name, 'opd', k, y_opd, y_pred)
    y_prob = model.predict_proba(x_opd)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_opd, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_opd, y_prob))
    df1.append(tmp)
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def RF_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'RFC'
    model = RandomForestClassifier(max_depth=7,
                                   n_estimators=110).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    tmp = CM_Calculator(model_name, 'train', k, y_train, y_pred)
    y_prob = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_train, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_val)
    tmp = CM_Calculator(model_name, 'val', k, y_val, y_pred)
    y_prob = model.predict_proba(x_val)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_val, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_test)
    tmp = CM_Calculator(model_name, 'test', k, y_test, y_pred)
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_test, y_prob))
    df1.append(tmp)
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    tmp = CM_Calculator(model_name, 'opd', k, y_opd, y_pred)
    y_prob = model.predict_proba(x_opd)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_opd, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_opd, y_prob))
    df1.append(tmp)
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def XGB_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'XGBC'
    model = xgb.XGBClassifier(max_depth=5,
                              learning_rate=0.01,
                              n_estimators=200,
                              use_label_encoder=False).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    tmp = CM_Calculator(model_name, 'train', k, y_train, y_pred)
    y_prob = model.predict_proba(x_train)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_train, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_val)
    tmp = CM_Calculator(model_name, 'val', k, y_val, y_pred)
    y_prob = model.predict_proba(x_val)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_val, y_prob))
    df1.append(tmp)
    
    y_pred = model.predict(x_test)
    tmp = CM_Calculator(model_name, 'test', k, y_test, y_pred)
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_test, y_prob))
    df1.append(tmp)
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    tmp = CM_Calculator(model_name, 'opd', k, y_opd, y_pred)
    y_prob = model.predict_proba(x_opd)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_opd, y_prob)
    tmp.append(metrics.auc(fpr, tpr))
    tmp.append(brier_score_loss(y_opd, y_prob))
    df1.append(tmp)
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def CM_Calculator(str1, str2, k, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)
    acc = (tp + tn) / (tp + fp + tn + fn)
    f1 = 2 * (ppv * tpr) / (ppv + tpr)
    lrp = tpr / (1 - tnr)
    lrn = (1 - tpr) / tnr
    dor = lrp / lrn
    tmp = ['%s_%s_%d'%(str1, str2, k), str1, str2, k, tp, fp, tn, fn, tpr, tnr, ppv, npv, acc, f1, lrp, lrn, dor]
    return tmp

def KP_Calculator(df1):
    df2 = []
    for i in list(df1.index):
        tmp = [df1[0][i]]
        y_true = []
        for j in range(1, len(df1.columns)):
            y_true.append(df1[j][i])
        for j in list(df1.index):
            y_pred = []
            for k in range(1, len(df1.columns)):
                y_pred.append(df1[k][j])
            tmp.append(cohen_kappa_score(y_true, y_pred))
        df2.append(tmp)
    df2 = pd.DataFrame(df2, columns=['Model']+list(df1[0]))
    return df2

def ENSB_Predictor(df1, df2, name):
    model_list = []
    for i in list(df1.index):
        if df1['Model'][i] not in model_list:
            model_list.append(df1['Model'][i])
    y_true = []
    for i in range(1, len(df2.columns)):
        y_true.append(df2[i][0])
    df3 = []
    df4 = []
    for i in model_list:
        tmp = []
        for j in range(1, len(df2.columns)):
            tmp.append(0)
        n = 0
        for j in list(df2.index):
            if df2[0][j] == 'Ans':
                continue
            elif df2[0][j].find('%s_'%i) != -1:
                n += 1
                for k in range(1, len(df2.columns)):
                    tmp[k-1] += df2[k][j]
        for k in range(0, len(tmp)):
            if tmp[k] < n / 2:
                tmp[k] = 0
            else:
                tmp[k] = 1
        df3.append(CM_Calculator(i, name, 0, y_true, tmp)+['-', '-'])
        df4.append(['%s_0'%i]+tmp)
    df1 = pd.concat([df1, pd.DataFrame(df3, columns=list(df1.columns))])
    df1.index = range(0, len(df1.index))
    df2 = pd.concat([df2, pd.DataFrame(df4, columns=list(df2.columns))])
    df2.index = range(0, len(df2.index))
    return df1, df2

def AAA(df2, df5):
    tmp = []
    df6 = df5.copy()
    y_true = []
    for i in range(1, len(df2.columns)):
        y_true.append(df2[i][0])
    for i in range(1, len(df2.index)):
        tmp.append([])
        tmp.append([])
        y_pred = []
        for j in range(1, len(df2.columns)):
            if df2[j][i] == y_true[j-1]:
                y_pred.append(1)
            else:
                y_pred.append(0)
        tmp[len(tmp)-2] = tmp[len(tmp)-1] + ['rslt-0', df2[0][i][:df2[0][i].index('_')], df2[0][i][df2[0][i].index('_')+1:], len(y_pred)]
        tmp[len(tmp)-1] = tmp[len(tmp)-1] + ['rslt-1', df2[0][i][:df2[0][i].index('_')], df2[0][i][df2[0][i].index('_')+1:], len(y_pred)]
        for j in list(df5.columns)[4:]:
            if j.find('SEX') != -1 or j.find('DG2_') != -1 or j.find('CCI_') != -1:
                cnt0, cnt1 = 0, 0
                cnt3, cnt4 = 0, 0
                for k in list(df5.index):
                    if df5[j][k] == 0:
                        if y_pred[k] == 1:
                            cnt0 += 1
                        cnt3 += 1
                    elif df5[j][k] == 1:
                        if y_pred[k] == 1:
                            cnt1 += 1
                        cnt4 += 1
                if cnt3 == 0:
                    cnt3 = 1
                if cnt4 == 0:
                    cnt4 = 1
                tmp[len(tmp)-2].append('%.3f'%(cnt0/cnt3))
                tmp[len(tmp)-1].append('%.3f'%(cnt1/cnt4))
            else:
                tmp[len(tmp)-2].append('')
                tmp[len(tmp)-1].append('')
        df6.loc[:, '%s'%df2[0][i]] = y_pred
    model_list = []
    for i in list(df2.index)[1:]:
        model_list.append(df2[0][i])
    model_list.sort()
    ft_name = list(df5.columns) + model_list
    df6 = df6[ft_name]
    return tmp, df6

if __name__ == '__main__':
    main()