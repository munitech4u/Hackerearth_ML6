import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
import os
from catboost import CatBoostClassifier
import time
import pickle


os.chdir("C:/Munish/hackerearth/earthquake")
trn=pd.read_csv("train.csv")
tst=pd.read_csv("test.csv")

i_len=trn.shape[0]

owner=pd.read_csv("Building_Ownership_Use.csv")
build_strct=pd.read_csv("Building_Structure.csv")

owner.drop(['district_id','vdcmun_id'],axis=1,inplace=True)
build_strct.drop(['district_id','vdcmun_id','ward_id'],axis=1,inplace=True)

trn_upd=pd.merge(trn,owner,how="left",on="building_id")
trn_upd=pd.merge(trn_upd,build_strct,how="left",on="building_id")


tst_upd=pd.merge(tst,owner,how="left",on="building_id")
tst_upd=pd.merge(tst_upd,build_strct,how="left",on="building_id")


comb=trn_upd.append(tst_upd).reset_index()

del owner,build_strct,trn,tst,trn_upd,tst_upd

comb['count_floors_post_eq'] = np.where(comb['count_floors_post_eq']>comb['count_floors_pre_eq'],comb['count_floors_pre_eq'], comb['count_floors_post_eq'])
comb['height_ft_post_eq'] = np.where(comb['height_ft_post_eq']>comb['height_ft_pre_eq'],comb['height_ft_pre_eq'], comb['height_ft_post_eq'])

comb['dist_cnt']=comb.groupby(['district_id'])['building_id'].transform('count')
comb['muncpl_cnt']=comb.groupby(['district_id','vdcmun_id'])['building_id'].transform('count')
comb['ward_cnt']=comb.groupby(['district_id','vdcmun_id','ward_id'])['building_id'].transform('count')


comb['dist_munc_ratio']=comb['muncpl_cnt']/comb['dist_cnt']
comb['muncpl_ward_ratio']=comb['ward_cnt']/comb['muncpl_cnt']
comb['new_var']=comb['dist_munc_ratio']*comb['muncpl_ward_ratio']

comb['dist_sum']=comb.groupby(['district_id'])['count_families'].transform('sum')
comb['muncpl_sum']=comb.groupby(['district_id','vdcmun_id'])['count_families'].transform('sum')
comb['ward_sum']=comb.groupby(['district_id','vdcmun_id','ward_id'])['count_families'].transform('sum')

comb['dist_munc_ratio1']=comb['muncpl_sum']/comb['dist_sum']
comb['muncpl_ward_ratio']=comb['ward_sum']/comb['muncpl_sum']

comb['has_repair_started']=comb['has_repair_started'].apply(lambda x:2 if pd.isnull(x) else x)



comb['height_diff']=comb['height_ft_pre_eq']-comb['height_ft_post_eq']
comb['floor_diff']=comb['count_floors_pre_eq']-comb['count_floors_post_eq']
   
comb['floor_ratio']=comb['count_floors_post_eq']/comb['count_floors_pre_eq']
comb['floor_ratio1']=comb['floor_diff']/comb['count_floors_pre_eq']
comb['height_ratio1']=comb['height_diff']/comb['height_ft_pre_eq']
comb['height_ratio2']=comb['height_ratio1']*comb['plinth_area_sq_ft']
comb['height_ratio3']=comb['height_ft_post_eq']/comb['height_ft_pre_eq']
comb['height_ratio4']=comb['height_ratio1']*comb['floor_diff']

comb['area_floor']=comb['plinth_area_sq_ft']*comb['floor_diff']
comb['area_floor1']=comb['plinth_area_sq_ft']*comb['height_diff']
comb['area_floor2']=comb['plinth_area_sq_ft']*comb['count_floors_pre_eq']
comb['area_floor3']=comb['plinth_area_sq_ft']*comb['count_floors_post_eq']


comb['area_plinth_pre']=comb['plinth_area_sq_ft']*comb['height_ft_pre_eq']
comb['area_plinth_post']=comb['plinth_area_sq_ft']*comb['height_ft_post_eq']
comb['area_plinth_ratio']=comb['area_plinth_post']*comb['area_plinth_pre']
comb['area_plinth_diff']=comb['area_plinth_pre']-comb['area_plinth_post']

comb['avg_height_pre']=comb['height_ft_pre_eq']/comb['count_floors_pre_eq']
comb['avg_height_post']=comb['height_ft_post_eq']/comb['count_floors_post_eq']
comb['avg_height_ratio']=comb['avg_height_post']/comb['avg_height_pre']

comb['floor_comb']=comb['ground_floor_type']+'_'+comb['other_floor_type']

comb['agelvl1']=comb.groupby(['district_id'])['age_building'].transform('std')
comb['agelvl2']=comb.groupby(['district_id','vdcmun_id'])['age_building'].transform('std')
comb['agelvl3']=comb.groupby(['district_id','vdcmun_id','ward_id'])['age_building'].transform('std')
    
tmp_df=comb[['district_id','vdcmun_id']].drop_duplicates().groupby('district_id').size().reset_index().rename(columns={0:'tot_munc'})
tmp_df1=comb[['district_id','ward_id']].drop_duplicates().groupby('district_id').size().reset_index().rename(columns={0:'tot_ward'})
tmp_df2=comb[['vdcmun_id','ward_id']].drop_duplicates().groupby('vdcmun_id').size().reset_index().rename(columns={0:'tot_munc_ward'})

comb=pd.merge(comb,tmp_df,how='left',on='district_id')
comb=pd.merge(comb,tmp_df1,how='left',on='district_id')
comb=pd.merge(comb,tmp_df2,how='left',on='vdcmun_id')


comb['avg_build']=comb['dist_cnt']/comb['tot_munc']
comb['avg_build1']=comb['muncpl_cnt']/comb['tot_ward']


comb['tot_risk']=(comb['has_geotechnical_risk']+comb['has_geotechnical_risk_fault_crack']+comb['has_geotechnical_risk_flood']+
          comb['has_geotechnical_risk_land_settlement']+comb['has_geotechnical_risk_landslide']+comb['has_geotechnical_risk_liquefaction']
          +comb['has_geotechnical_risk_other']+comb['has_geotechnical_risk_rock_fall'])


comb['tot_struct']=(comb['has_superstructure_adobe_mud']+comb['has_superstructure_mud_mortar_stone']+
                    comb['has_superstructure_stone_flag']+comb['has_superstructure_mud_mortar_brick']+comb['has_superstructure_cement_mortar_brick']+
                    comb['has_superstructure_timber']+comb['has_superstructure_bamboo']+comb['has_superstructure_rc_non_engineered']+
                    comb['has_superstructure_rc_engineered']+comb['has_superstructure_other'])
gc.collect()


flots=[col for col in comb.columns if comb[col].dtype == 'float64']
ints=[col for col in comb.columns if comb[col].dtype == 'int64']

for i in range(len(flots)):
    comb[flots[i]]=comb[flots[i]].astype('float32')
    
for i in range(len(ints)):
    comb[ints[i]]=comb[ints[i]].astype('int32')
    
cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','vdcmun_id','ward_id','has_repair_started']

for i in range(len(cats)):
    comb[cats[i]]=comb[cats[i]].astype('category')

del tmp_df,tmp_df1,tmp_df2
   
gc.collect()


train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]
del comb

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()

folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

train_df['label']=train_df['damage_grade'].map(d)
sub_preds = np.zeros([test_df.shape[0],5])
i=0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[v1], train_df['label'])):
    train_x, train_y = train_df[v1].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[v1].iloc[valid_idx], train_df['label'].iloc[valid_idx]
    i+=1
    print ('iteration-%s'%i) 
    clf = clf = LGBMClassifier(
    boosting_type= 'gbdt',
    objective= 'multiclass',
    num_class=5,
    metric= 'multi_logloss',
    learning_rate= 0.05,
    max_depth= 7,
    num_leaves= 60,
    feature_fraction= 0.7,
    bagging_fraction= 1,
    bagging_freq= 20,
#    min_data_in_leaf=100,
    nthread=4,
    n_estimators=5000)

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            verbose= 100, early_stopping_rounds= 100)

    sub_preds += clf.predict_proba(test_df[v1], num_iteration=clf.best_iteration_)/ folds.n_splits

with open('lgb_prob_less_features.pkl', 'wb') as output:
    pickle.dump(sub_preds, output)
    
predictions = []

for x in sub_preds:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)



