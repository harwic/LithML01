#   注意！本程式需"dataset_1_psm資料檔"，執行"pgm_07_A.sas"取得。
import os
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pylab as plt
import shap as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import accuracy_score
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
    case_name = '1-DM-7-1-RankList'
    op_path = '%s\\PY_output\\rslt-1\\rslt-%s.xlsx'%(PATH, case_name)
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
    
    #   標準化資料集
    X = dataset.copy()
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
            
    #   切資料集
    tmp_indx, test_indx, opd_indx = [], [], []
    for i in list(X.index):
        if X['GP'][i] == -1:
            opd_indx.append(i)
        elif X['FLAG'][i].find('NO') != -1:
            tmp_indx.append(i)
        else:
            test_indx.append(i)
    X = X.drop(columns=['LABVAL', 'No', 'GP', 'FLAG', '_PS_', '_MATCHWGT_'])
    df = pd.read_excel(r'C:\Users\harwi\Dropbox\!!01與曉菁共享資料夾\202001_MOST\04-Stage1\PY_output\codebook_1.xlsx', engine='openpyxl', sheet_name='3')
    tmp = []
    for i in list(X.columns):
        for j in list(df['AAA']):
            if i == j:
                tmp.append(list(df['Name_1'])[list(df['AAA']).index(j)])
                break
    X.columns = tmp
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
    SKF = StratifiedKFold(n_splits=5)
    train_indx, val_indx = [], []
    for indx1, indx2 in SKF.split(tmp_X, tmp_Y):
        train_indx.append(indx1.tolist())
        val_indx.append(indx2.tolist())
    
    #   特徵排名
    df1 = LASSO_Ranker(tmp_X, tmp_Y, model_path)
    df1.loc[:, 'SVC-F'] = SVM_Ranker_1(tmp_X, tmp_Y, x_test, y_test, train_indx, val_indx)
    df1.loc[:, 'SVC-B'] = SVM_Ranker_2(tmp_X, tmp_Y, x_test, y_test, train_indx, val_indx)
    df = RF_Ranker(tmp_X, tmp_Y, model_path)
    df1 = pd.concat([df1, df], axis=1)
    df = XGB_Ranker(tmp_X, tmp_Y, model_path)
    df1 = pd.concat([df1, df], axis=1)
    df1 = AAA(df1, list(tmp_X.columns))
    df1 = BBB(df1, list(tmp_X.columns))
    with pd.ExcelWriter(op_path, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Rank', index=False)
    writer.close()
    return

def SVM_Ranker_1(X, Y, x_test, y_test, train_indx, val_indx):
    ft_list = list(X.columns)
    rk_list = []
    best_acc = 0
    best_ft = ''
    for i in ft_list:
        tmp_X = X[[i]]
        tmp_x_test = x_test[[i]]
        acc = []
        for j in range(0, len(train_indx)):
            x_train = tmp_X.drop(index=val_indx[j])
            x_train.index = range(0, len(train_indx[j]))
            y_train = Y.drop(index=val_indx[j])
            y_train.index = range(0, len(train_indx[j]))
            y_pred = SVC(C=1).fit(x_train, y_train).predict(tmp_x_test)
            acc.append(accuracy_score(y_test, y_pred))
            
        acc = np.array(acc).mean()
        if acc > best_acc:
            best_acc = acc
            best_ft = i
    basic_acc = best_acc
    rk_list.append(best_ft)
    ft_list.remove(best_ft)
    while len(ft_list) > 0:
        best_acc1 = -1.1
        best_acc2 = 0
        best_ft = ''
        for i in ft_list:
            used_ft = rk_list.copy()
            acc = []
            if len(ft_list) != 1:
                used_ft.append(i)
                tmp_X = X[used_ft]
                tmp_x_test = x_test[used_ft]
            else:
                best_ft = i
                break
            for j in range(0, len(train_indx)):
                x_train = tmp_X.drop(index=val_indx[j])
                x_train.index = range(0, len(train_indx[j]))
                y_train = Y.drop(index=val_indx[j])
                y_train.index = range(0, len(train_indx[j]))
                y_pred = SVC(C=1).fit(x_train, y_train).predict(tmp_x_test)
                acc.append(accuracy_score(y_test, y_pred))
                
            acc = np.array(acc).mean()
            if acc - basic_acc > best_acc1:
                best_acc1 = acc - basic_acc
                best_acc2 = acc
                best_ft = i
        basic_acc = best_acc2
        rk_list.append(best_ft)
        ft_list.remove(best_ft)
    return rk_list

def SVM_Ranker_2(X, Y, x_test, y_test, train_indx, val_indx):
    ft_list = list(X.columns)
    basic_acc = []
    rk_list = []
    for i in range(0, len(train_indx)):
        x_train = X.drop(index=val_indx[i])
        x_train.index = range(0, len(train_indx[i]))
        y_train = Y.drop(index=val_indx[i])
        y_train.index = range(0, len(train_indx[i]))
        y_pred = SVC(C=1).fit(x_train, y_train).predict(x_test)
        basic_acc.append(accuracy_score(y_test, y_pred))
        
    basic_acc = np.array(basic_acc).mean()
    while len(ft_list) > 0:
        best_acc1 = -1.1
        best_acc2 = 0
        best_ft = ''
        for i in ft_list:
            used_ft = ft_list.copy()
            acc = []
            if len(ft_list) != 1:
                used_ft.remove(i)
                tmp_X = X[used_ft]
                tmp_x_test = x_test[used_ft]
            else:
                best_ft = i
                break
            for j in range(0, len(train_indx)):
                x_train = tmp_X.drop(index=val_indx[j])
                x_train.index = range(0, len(train_indx[j]))
                y_train = Y.drop(index=val_indx[j])
                y_train.index = range(0, len(train_indx[j]))
                y_pred = SVC(C=1).fit(x_train, y_train).predict(tmp_x_test)
                acc.append(accuracy_score(y_test, y_pred))
                
            acc = np.array(acc).mean()
            if acc - basic_acc > best_acc1:
                best_acc1 = acc - basic_acc
                best_acc2 = acc
                best_ft = i
        basic_acc = best_acc2
        if len(rk_list) == 0:
            rk_list.append(best_ft)
        else:
            rk_list.insert(0, best_ft)
        ft_list.remove(best_ft)
    return rk_list

def RF_Ranker(x_train, y_train, model_path):
    model_name = 'RFC'
    model = RandomForestClassifier(max_depth=7,
                                   n_estimators=110).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s.joblib'%(model_path, model_name))
    explainer = sp.TreeExplainer(model)
    sp_values = explainer(x_train)
    scores = np.abs(sp_values.values[:, :, 1]).mean(0)
    df = []
    for i in list(x_train.columns):
        df.append([i, scores[len(df)]])
    df = pd.DataFrame(df, columns=['%s'%model_name, 'Score']).sort_values(by=['Score'], ascending=False)
    df.index = range(0, len(df.index))
    df = df.drop(columns=['Score'])
    tmp = sp.Explanation(sp_values[:, :, 1], data=x_train, feature_names=list(x_train.columns))
    sp.summary_plot(tmp, max_display=10,)
    sp.summary_plot(tmp, max_display=20)
    sp.summary_plot(tmp, max_display=len(x_train.columns))
    sp.summary_plot(tmp, max_display=10, plot_type='bar', color='green')
    sp.summary_plot(tmp, max_display=20, plot_type='bar', color='green')
    sp.summary_plot(tmp, max_display=len(x_train.columns), plot_type='bar', color='green')
    return df

def XGB_Ranker(x_train, y_train, model_path):
    model_name = 'XGBC'
    model = xgb.XGBClassifier(max_depth=5,
                              learning_rate=0.01,
                              n_estimators=200,
                              use_label_encoder=False).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s.joblib'%(model_path, model_name))
    explainer = sp.TreeExplainer(model)
    sp_values = explainer(x_train)
    scores = np.abs(sp_values.values).mean(0)
    df = []
    for i in list(x_train.columns):
        df.append([i, scores[len(df)]])
    df = pd.DataFrame(df, columns=['%s'%model_name, 'Score']).sort_values(by=['Score'], ascending=False)
    df.index = range(0, len(df.index))
    df = df.drop(columns=['Score'])
    sp.summary_plot(sp_values, max_display=10)
    sp.summary_plot(sp_values, max_display=20)
    sp.summary_plot(sp_values, max_display=len(x_train.columns))
    sp.summary_plot(sp_values, max_display=10, plot_type='bar', color='green')
    sp.summary_plot(sp_values, max_display=20, plot_type='bar', color='green')
    sp.summary_plot(sp_values, max_display=len(x_train.columns), plot_type='bar', color='green')
    return df

def LASSO_Ranker(x_train, y_train, model_path):
    model_name = 'LogR'
    model = LassoLarsCV(precompute=False).fit(x_train, y_train)
    joblib.dump(model, '%s\\%s.joblib'%(model_path, model_name))
    m_log_alphas = -np.log10(model.alphas_)
    tmp = []
    coef_path = model.coef_path_.tolist()
    for i in range(0, len(coef_path[0])):
        for j in range(0, len(coef_path)):
            if coef_path[j][len(coef_path[0])-i-1] == 0 and list(x_train.columns)[j] not in tmp:
                tmp.append(list(x_train.columns)[j])
    df1 = pd.DataFrame(tmp, columns=['LogR-X'])
    tmp = []
    for i in range(0, len(coef_path)):
        tmp.append([list(x_train.columns)[i], abs(coef_path[i][len(coef_path[0])-1])])
    df2 = pd.DataFrame(tmp, columns=['LogR-Y', 'S']).sort_values(by=['S'])
    df2.index = range(0, len(df2.index))
    df2 = df2.drop(columns=['S'])
    df3 = pd.concat([df1, df2], axis=1)
    df4 = df3.reindex(index=df3.index[::-1])
    df4.index = range(0, len(df4.index))
    fig = plt.figure(figsize=(8, 6))
    c_list = ['r', 'g', 'b', 'c', 'y']
    ls_list = ['-', '-.']
    j = 0
    plt.plot(m_log_alphas, [0]*len(model.coef_path_[i]), c='k', zorder=len(df4.index)-10, ls='--')
    for i in list(df4['LogR-X'][:]):
        if j < 10:
            plt.plot(m_log_alphas, model.coef_path_[list(x_train.columns).index(i)], label='%s'%i, c=c_list[j//2], ls=ls_list[j%2], zorder=len(df4.index)-j)
        elif j == 10:
            plt.plot(m_log_alphas, model.coef_path_[list(x_train.columns).index(i)], label='Other', c='gray', zorder=len(df4.index)-11)
        else:
            plt.plot(m_log_alphas, model.coef_path_[list(x_train.columns).index(i)], c='#d2d2d2', zorder=len(df4.index)-11)
        j += 1
    plt.legend(bbox_to_anchor=(1.0, 1))
    plt.xlabel('Log alpha', fontweight='bold')
    plt.ylabel('Coefficient', fontweight='bold')
    plt.show()
    return df4

def AAA(df1, ft_list):
    rk_list = []
    for i in ft_list:
        rk_list.append([i, 0, -1])
    for i in list(df1.index):
        rk_list[ft_list.index(df1['LogR-X'][i])][1] += 1
        rk_list[ft_list.index(df1['SVC-F'][i])][1] += 1
        rk_list[ft_list.index(df1['RFC'][i])][1] += 1
        rk_list[ft_list.index(df1['XGBC'][i])][1] += 1
        for j in range(0, len(rk_list)):
            if rk_list[j][1] == 4 and rk_list[j][2] == -1:
                rk_list[j][2] = i
    df2 = pd.DataFrame(rk_list, columns=['All-F', 'B', 'C']).sort_values(by=['C']).drop(columns=['B'])
    df2.index = range(0, len(df2.index))
    tmp = []
    new_rank = []
    for i in list(df2.index):
        if df2['C'][i] not in tmp:
            tmp.append(df2['C'][i])
        new_rank.append(len(tmp))
    df2 = df2.drop(columns=['C'])
    df1 = pd.concat([df1, df2], axis=1)
    return df1

def BBB(df1, ft_list):
    rk_list = []
    for i in ft_list:
        rk_list.append([i, 0, -1])
    for i in list(df1.index):
        rk_list[ft_list.index(df1['LogR-X'][i])][1] += 1
        rk_list[ft_list.index(df1['SVC-B'][i])][1] += 1
        rk_list[ft_list.index(df1['RFC'][i])][1] += 1
        rk_list[ft_list.index(df1['XGBC'][i])][1] += 1
        for j in range(0, len(rk_list)):
            if rk_list[j][1] == 4 and rk_list[j][2] == -1:
                rk_list[j][2] = i
    df2 = pd.DataFrame(rk_list, columns=['All-B', 'B', 'C']).sort_values(by=['C']).drop(columns=['B'])
    df2.index = range(0, len(df2.index))
    tmp = []
    new_rank = []
    for i in list(df2.index):
        if df2['C'][i] not in tmp:
            tmp.append(df2['C'][i])
        new_rank.append(len(tmp))
    df2 = df2.drop(columns=['C'])
    df1 = pd.concat([df1, df2], axis=1)
    return df1

if __name__ == '__main__':
    main()