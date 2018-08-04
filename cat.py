# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:22:43 2018

@author: vvootl010
"""

train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]

cats = [col for col in train_df.columns if train_df[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','vdcmun_id','ward_id']

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()

cat_cols=[i for i,x in enumerate(v1) if x in cats]

train_x, valid_x, train_y, valid_y = train_test_split(train_df[v1], label, test_size=0.30, random_state=42)

model = CatBoostClassifier(iterations=5000, learning_rate=0.05, depth=6, loss_function='MultiClass',od_type='Iter')

model.fit(train_x, 
          train_y, 
          cat_features=cat_cols,
          use_best_model=True,
          eval_set=(valid_x,valid_y)  
          )

preds = model.predict(test_df[v1])

pred=preds.ravel()
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':pred})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub_cat.csv",index=False)


with open('model_catb.pkl', 'wb') as output:
    pickle.dump(model, output)

prob_catb = model.predict_proba(test_df[v1])
    
with open('catboost_prob.pkl', 'wb') as output:
    pickle.dump(prob_catb, output)
    
preds = model.predict(train_df[v1]).ravel()

x=pd.DataFrame({'building_id':train_df['building_id'],'grade':preds,'district':train_df['district_id'],
                'actual_damage':train_df['damage_grade']})

x['pred_damage']=x['grade'].map(d1)

y=x.groupby(['actual_damage','pred_damage'])['building_id'].count().reset_index()



