import os
import joblib
import math
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def main():
    PATH = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    dataset_path = '%s\\SAS_output\\dataset_1_psm.sas7bdat'%PATH
    if not os.path.isabs(dataset_path):
        print('Error: %s no found.'%dataset_path)
        return
    dataset = pd.read_sas(dataset_path, encoding='iso-8859-1')
    case_name = '1-CM-1-All'
    model_path = '%s\\PY_output\\model\\%s'%(PATH, case_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        pass
    
    #   Filter data
    drop_list = []
    for i in list(dataset.index):
        if dataset['GP'][i] > 1:
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
    
    #   Set Outcome
    Y = dataset['LABVAL']
    
    #   Normalization & Split
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
    
    #   Establish model
    KF = KFold(n_splits=5)
    k = 1
    df1 = []
    df2 = [['Ans']+list(y_test)]
    df3 = [['Ans']+list(y_opd)]
    for train_indx, val_indx in KF.split(tmp_X, tmp_Y):
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
    ft_list = ['Name', 'Model', 'Set', 'K', 'Acc.05', 'Acc.10', 'Acc.15', 'Acc.20', 'MSE', 'RMSE', 'MAE', 'SDAE(n)', 'SDAE(n-1)', 'R2', 'A-R2']
    df1 = pd.DataFrame(df1, columns=ft_list)
    df2 = pd.DataFrame(df2, columns=list(range(0, len(df2[0]))))
    df3 = pd.DataFrame(df3, columns=list(range(0, len(df3[0]))))
    df1, df2 = ENSB_Predictor(df1, df2, x_test, 'ens_test')
    df1, df3 = ENSB_Predictor(df1, df3, x_opd, 'ens_opd')
    df5 = x_test.copy()
    df5.loc[:, 'Type'] = 'data'
    df5.loc[:, 'Model'] = ''
    df5.loc[:, 'K'] = ''
    df5.loc[:, 'N'] = ''
    tmp = ['Type', 'Model', 'K', 'N'] + list(x_test.columns)
    df5 = df5[tmp]
    file_path = '%s\\PY_output\\rslt-1\\rslt-%s.xlsx'%(PATH, case_name)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Score', index=False)
        df2.to_excel(writer, sheet_name='Pred_test', index=False)
        df3.to_excel(writer, sheet_name='Pred_opd', index=False)
        df5.to_excel(writer, sheet_name='Dataset_test', index=False)
    writer.close()
    return

def LR_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'LinR'
    model = LinearRegression(positive=True).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    df1.append(EM_Calculator(model_name, 'train', k, y_train, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_val)
    df1.append(EM_Calculator(model_name, 'val', k, y_val, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_test)
    df1.append(EM_Calculator(model_name, 'test', k, y_test, y_pred, len(x_test.columns)))
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    df1.append(EM_Calculator(model_name, 'opd', k, y_opd, y_pred, len(x_opd.columns)))
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def SVM_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'SVR'
    model = SVR(C=1).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    df1.append(EM_Calculator(model_name, 'train', k, y_train, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_val)
    df1.append(EM_Calculator(model_name, 'val', k, y_val, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_test)
    df1.append(EM_Calculator(model_name, 'test', k, y_test, y_pred, len(x_test.columns)))
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    df1.append(EM_Calculator(model_name, 'opd', k, y_opd, y_pred, len(x_opd.columns)))
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def RF_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'RFR'
    model = RandomForestRegressor(max_depth=7,
                                  n_estimators=110).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    df1.append(EM_Calculator(model_name, 'train', k, y_train, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_val)
    df1.append(EM_Calculator(model_name, 'val', k, y_val, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_test)
    df1.append(EM_Calculator(model_name, 'test', k, y_test, y_pred, len(x_test.columns)))
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    df1.append(EM_Calculator(model_name, 'opd', k, y_opd, y_pred, len(x_opd.columns)))
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def XGB_Model(x_train, x_val, x_test, x_opd, y_train, y_val, y_test, y_opd, k, df1, df2, df3, model_path):
    model_name = 'XGBR'
    model = xgb.XGBRegressor(max_depth=5,
                             learning_rate=0.01,
                             n_estimators=200).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s_%d.joblib'%(model_path, model_name, k))
    
    y_pred = model.predict(x_train)
    df1.append(EM_Calculator(model_name, 'train', k, y_train, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_val)
    df1.append(EM_Calculator(model_name, 'val', k, y_val, y_pred, len(x_test.columns)))
    
    y_pred = model.predict(x_test)
    df1.append(EM_Calculator(model_name, 'test', k, y_test, y_pred, len(x_test.columns)))
    df2.append(['%s_%d'%(model_name, k)]+list(y_pred))
    
    y_pred = model.predict(x_opd)
    df1.append(EM_Calculator(model_name, 'opd', k, y_opd, y_pred, len(x_opd.columns)))
    df3.append(['%s_%d'%(model_name, k)]+list(y_pred))
    return df1, df2, df3

def EM_Calculator(str1, str2, k, y_true, y_pred, p):
    acc = [0, 0, 0, 0]
    for i in list(y_true.index):
        for j in range(0, 4):
            if abs(y_true[i] - y_pred[i]) <= 0.05 + (0.05 * j):
                acc[j] += 1
    for i in range(0, len(acc)):
        acc[i] /= len(y_true.index)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    tmp = []
    for i in list(y_true.index):
        tmp.append(abs(y_true[i] - y_pred[i]))
    sdae0 = np.std(np.array(tmp), ddof=0)
    sdae1 = np.std(np.array(tmp), ddof=1)
    r2 = r2_score(y_true, y_pred)
    ar2 = 1 - (1 - r2) * (len(y_true.index) - 1) / (len(y_true.index) - p - 1)
    tmp = ['%s_%s_%d'%(str1, str2, k), str1, str2, k, acc[0], acc[1], acc[2], acc[3], mse, rmse, mae, sdae0, sdae1, r2, ar2]
    return tmp

def ENSB_Predictor(df1, df2, x_test, name):
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
            tmp.append([])
        for j in list(df2.index):
            if df2[0][j] == 'Ans':
                continue
            elif df2[0][j].find('%s_'%i) != -1:
                for k in range(1, len(df2.columns)):
                    tmp[k-1].append(df2[k][j])
        y_pred = []
        for k in range(0, len(tmp)):
            tmp[k].sort()
            y_pred.append(np.median(tmp[k]))
        df3.append(EM_Calculator(i, name, 0, pd.Series(y_true), y_pred, len(x_test.columns)))
        df4.append(['%s_0'%i]+y_pred)
    df1 = pd.concat([df1, pd.DataFrame(df3, columns=list(df1.columns))])
    df1.index = range(0, len(df1.index))
    df2 = pd.concat([df2, pd.DataFrame(df4, columns=list(df2.columns))])
    df2.index = range(0, len(df2.index))
    return df1, df2

if __name__ == '__main__':
    main()