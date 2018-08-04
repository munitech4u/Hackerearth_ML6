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


#trn['district_id'].value_counts()
#trn['building_id'].nunique()

owner.drop(['district_id','vdcmun_id'],axis=1,inplace=True)
build_strct.drop(['district_id','vdcmun_id','ward_id'],axis=1,inplace=True)

trn_upd=pd.merge(trn,owner,how="left",on="building_id")
trn_upd=pd.merge(trn_upd,build_strct,how="left",on="building_id")


tst_upd=pd.merge(tst,owner,how="left",on="building_id")
tst_upd=pd.merge(tst_upd,build_strct,how="left",on="building_id")


#df = trn_upd.append(tst_upd).reset_index()

#x=tst_upd[~tst_upd['ward_id'].isin(trn_upd['ward_id'])]
#x=trn_upd[~trn_upd['ward_id'].isin(tst_upd['ward_id'])]

#y=list(set(x['district_id']).intersection(set(trn_upd['district_id'])))

#trn[trn['district_id'].isin(y)].shape

#trn1=trn_upd[trn_upd['ward_id'].isin(tst_upd['ward_id'])]
#tst1=tst_upd[tst_upd['ward_id'].isin(trn1['ward_id'])]

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


ls1=['has_geotechnical_risk','has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood','has_geotechnical_risk_land_settlement',
     'has_geotechnical_risk_landslide','has_geotechnical_risk_liquefaction','has_geotechnical_risk_other','has_geotechnical_risk_rock_fall']

for i in range(len(ls1)):
    comb[str('geolvl1'+str(i))]=comb.groupby(['district_id'])[ls1[i]].transform('mean')
    comb[str('geolvl2'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls1[i]].transform('mean')
    comb[str('geolvl3'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls1[i]].transform('mean')
    comb[str('geosub'+str(i))]=comb[str('geolvl1'+str(i))]-comb[str('geolvl2'+str(i))]
    comb[str('geosub2'+str(i))]=comb[str('geolvl1'+str(i))]-comb[str('geolvl3'+str(i))]
    comb[str('geosub3'+str(i))]=comb[str('geolvl2'+str(i))]-comb[str('geolvl3'+str(i))]
    comb[str('geolvl4'+str(i))]=comb.groupby(['district_id'])[ls1[i]].transform('sum')
    comb[str('geolvl5'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls1[i]].transform('sum')
    comb[str('geolvl6'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls1[i]].transform('sum')


ls2=['has_secondary_use','has_secondary_use_agriculture','has_secondary_use_hotel','has_secondary_use_rental',
     'has_secondary_use_institution','has_secondary_use_school','has_secondary_use_industry','has_secondary_use_health_post','has_secondary_use_gov_office',
     'has_secondary_use_use_police','has_secondary_use_other']
   
for i in range(len(ls2)):
    comb[str('uselvl1'+str(i))]=comb.groupby(['district_id'])[ls2[i]].transform('mean')
    comb[str('uselvl2'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls2[i]].transform('mean')
    comb[str('uselvl3'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls2[i]].transform('mean')
    comb[str('usesub'+str(i))]=comb[str('uselvl1'+str(i))]-comb[str('uselvl2'+str(i))]
    comb[str('usesub2'+str(i))]=comb[str('uselvl1'+str(i))]-comb[str('uselvl3'+str(i))]
    comb[str('usesub3'+str(i))]=comb[str('uselvl2'+str(i))]-comb[str('uselvl3'+str(i))]
    comb[str('uselvl4'+str(i))]=comb.groupby(['district_id'])[ls2[i]].transform('sum')
    comb[str('uselvl5'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls2[i]].transform('sum')
    comb[str('uselvl6'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls2[i]].transform('sum')

ls3=['has_superstructure_adobe_mud','has_superstructure_mud_mortar_stone','has_superstructure_stone_flag','has_superstructure_mud_mortar_brick',
     'has_superstructure_cement_mortar_brick','has_superstructure_timber','has_superstructure_bamboo','has_superstructure_rc_non_engineered','has_superstructure_rc_engineered',
	 'has_superstructure_other']
   
for i in range(len(ls3)):
    comb[str('structlvl1'+str(i))]=comb.groupby(['district_id'])[ls3[i]].transform('mean')
    comb[str('structlvl2'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls3[i]].transform('mean')
    comb[str('structlvl3'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls3[i]].transform('mean')
    comb[str('structlvl4'+str(i))]=comb.groupby(['district_id'])[ls3[i]].transform('sum')
    comb[str('structlvl5'+str(i))]=comb.groupby(['district_id','vdcmun_id'])[ls3[i]].transform('sum')
    comb[str('structlvl6'+str(i))]=comb.groupby(['district_id','vdcmun_id','ward_id'])[ls3[i]].transform('sum')


#agg=['min','max','mean']

#for i in range(len(ls3)):
#    for j in range(len(agg)):
#       comb[str('agg_var1'+str(i)+str(j))]=comb.groupby(['district_id','vdcmun_id','ward_id','foundation_type']+[ls3[i]])['plinth_area_sq_ft'].transform(agg[j])
#       comb[str('agg_var2'+str(i)+str(j))]=comb.groupby(['district_id','vdcmun_id','ward_id','foundation_type']+[ls3[i]])['height_ft_pre_eq'].transform(agg[j])
#       comb[str('agg_var3'+str(i)+str(j))]=comb.groupby(['district_id','vdcmun_id','ward_id','foundation_type']+[ls3[i]])['height_ft_post_eq'].transform(agg[j])
#       comb[str('agg_var_sub'+str(i)+str(j))]=comb[str('agg_var3'+str(i)+str(j))]-comb[str('agg_var2'+str(i)+str(j))]

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

#comb['ht_ratio_ward1']=comb.groupby(['ward_id','count_floors_pre_eq','foundation_type'])['height_ratio1'].transform('mean')
#comb['ht_ratio_ward2']=comb.groupby(['ward_id','count_floors_pre_eq','foundation_type'])['height_ratio1'].transform('min')
#comb['ht_ratio_ward3']=comb.groupby(['ward_id','count_floors_pre_eq','foundation_type'])['height_ratio1'].transform('max')
#
#comb['ht_ratio_ward4']=comb.groupby(['ward_id'])['height_ratio1'].transform('mean')
#comb['ht_ratio_ward5']=comb.groupby(['ward_id'])['height_ratio1'].transform('min')
#comb['ht_ratio_ward6']=comb.groupby(['ward_id'])['height_ratio1'].transform('max')
#
#comb['ht_ratio_ward7']=comb.groupby(['vdcmun_id','area_assesed'])['height_ratio1'].transform('mean')
#comb['ht_ratio_ward8']=comb.groupby(['vdcmun_id','area_assesed'])['height_ratio1'].transform('min')
#comb['ht_ratio_ward9']=comb.groupby(['vdcmun_id','area_assesed'])['height_ratio1'].transform('max')

comb['floor_comb']=comb['ground_floor_type']+'_'+comb['other_floor_type']

comb['agelvl1']=comb.groupby(['district_id'])['age_building'].transform('mean')
comb['agelvl2']=comb.groupby(['district_id','vdcmun_id'])['age_building'].transform('mean')
comb['agelvl3']=comb.groupby(['district_id','vdcmun_id','ward_id'])['age_building'].transform('mean')
comb['agesub1']=comb['agelvl1']-comb['agelvl2']
comb['agesub2']=comb['agelvl2']-comb['agelvl3']
comb['agesub3']=comb['agelvl1']-comb['agelvl3']
    
tmp_df=comb[['district_id','vdcmun_id']].drop_duplicates().groupby('district_id').size().reset_index().rename(columns={0:'tot_munc'})
tmp_df1=comb[['district_id','ward_id']].drop_duplicates().groupby('district_id').size().reset_index().rename(columns={0:'tot_ward'})
tmp_df2=comb[['vdcmun_id','ward_id']].drop_duplicates().groupby('vdcmun_id').size().reset_index().rename(columns={0:'tot_munc_ward'})

comb=pd.merge(comb,tmp_df,how='left',on='district_id')
comb=pd.merge(comb,tmp_df1,how='left',on='district_id')
comb=pd.merge(comb,tmp_df2,how='left',on='vdcmun_id')


comb['avg_build']=comb['dist_cnt']/comb['tot_munc']
comb['avg_build1']=comb['muncpl_cnt']/comb['tot_ward']

#comb['ward_id'], uniques=pd.factorize(comb['ward_id'])

comb['tot_risk']=(comb['has_geotechnical_risk']+comb['has_geotechnical_risk_fault_crack']+comb['has_geotechnical_risk_flood']+
          comb['has_geotechnical_risk_land_settlement']+comb['has_geotechnical_risk_landslide']+comb['has_geotechnical_risk_liquefaction']
          +comb['has_geotechnical_risk_other']+comb['has_geotechnical_risk_rock_fall'])


comb['tot_struct']=(comb['has_superstructure_adobe_mud']+comb['has_superstructure_mud_mortar_stone']+
                    comb['has_superstructure_stone_flag']+comb['has_superstructure_mud_mortar_brick']+comb['has_superstructure_cement_mortar_brick']+
                    comb['has_superstructure_timber']+comb['has_superstructure_bamboo']+comb['has_superstructure_rc_non_engineered']+
                    comb['has_superstructure_rc_engineered']+comb['has_superstructure_other'])
gc.collect()

tmp_df3=comb.groupby(['vdcmun_id','damage_grade'])['building_id'].count().unstack().reset_index()

ls3=list(tmp_df3.columns)
for i in range(len(ls3)):
    tmp_df3[ls3[i]]=tmp_df3[ls3[i]].fillna(0)

for i in range(len(ls3)):
    tmp_df3['g'+str(i)]=tmp_df3[ls3[i]]/(tmp_df3[ls3[1]]+tmp_df3[ls3[2]]+tmp_df3[ls3[3]]+tmp_df3[ls3[4]]+tmp_df3[ls3[5]])

comb=pd.merge(comb,tmp_df3[['vdcmun_id','g1','g2','g3','g4','g5']],how='left',on='vdcmun_id')


flots=[col for col in comb.columns if comb[col].dtype == 'float64']
ints=[col for col in comb.columns if comb[col].dtype == 'int64']

for i in range(len(flots)):
    comb[flots[i]]=comb[flots[i]].astype('float32')
    
for i in range(len(ints)):
    comb[ints[i]]=comb[ints[i]].astype('int32')
    
cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','vdcmun_id','ward_id','has_repair_started']

#comb = pd.get_dummies(comb, columns= cats, dummy_na=True)

for i in range(len(cats)):
    comb[cats[i]]=comb[cats[i]].astype('category')

del tmp_df,tmp_df1,tmp_df2,tmp_df3
   
gc.collect()


train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]
del comb

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()

train_data = lgbm.Dataset(train_df[v1].values, label=label)

params = {'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class':5,
    'metric': 'multi_logloss',
    'learning_rate': 0.06,
    'max_depth': 8,
    'num_leaves': 35,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 20,
    'nthread':4}

#lgb_cv = lgbm.cv(params, train_data, num_boost_round=2000, nfold=3, shuffle=True, stratified=True, verbose_eval=20, early_stopping_rounds=100)

#nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
#print(nround)

start=time.time()
model_lgb = lgbm.train(params, train_data, num_boost_round=5000)
print("Took " + str(round(((time.time())-start)/60)) + " minutes")

imp = pd.DataFrame()
imp["feature"] = v1
imp["importance"] = model_lgb.feature_importance

##################fitting cross validation
train_x, valid_x, train_y, valid_y = train_test_split(train_df[v1], label, test_size=0.40, random_state=42)

clf = LGBMClassifier(
       boosting_type= 'gbdt',
    objective= 'multiclass',
    num_class=5,
    metric= 'multi_logloss',
    learning_rate= 0.06,
    max_depth= 8,
    num_leaves= 50,
    feature_fraction= 0.6,
    bagging_fraction= 0.6,
    bagging_freq= 20,
#   min_data_in_leaf=100,
    nthread=4,
    n_estimators=5500)

start=time.time()
clf.fit(train_df[v1], label, verbose= 100, early_stopping_rounds= 200,eval_set=[(train_df[v1], label), (train_df[v1],label)])
print("Took " + str(round(((time.time())-start)/60)) + " minutes")


clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        verbose= 100, early_stopping_rounds= 200)

imp = pd.DataFrame()
imp["feature"] = v1
imp["importance"] = clf.feature_importances_

start=time.time()
preds = clf.predict(test_df[v1])
print("Took " + str(round(((time.time())-start)/60)) + " minutes")

###############predictions
start=time.time()
preds = model_lgb.predict(test_df[v1])
print("Took " + str(round(((time.time())-start)/60)) + " minutes")


predictions = []

for x in preds:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':preds})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)




