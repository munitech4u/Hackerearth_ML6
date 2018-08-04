with open(r"catboost_prob.pkl", "rb") as input_file:
           prob_catb = pickle.load(input_file)
           
with open(r"catboost_prob_less_feat.pkl", "rb") as input_file:
           prob_catb_less_feat = pickle.load(input_file)
           
#with open(r"catboost_prob_wout_munc.pkl", "rb") as input_file:
#           prob_catb_wout_munc = pickle.load(input_file)
           
#with open(r"keras_prob_all.pkl", "rb") as input_file:
#           keras_prob = pickle.load(input_file)
           
with open(r"lgb_prob_without_ward.pkl", "rb") as input_file:
           prob_lgb_ward = pickle.load(input_file)
           
with open(r"lgb_prob_less_features.pkl", "rb") as input_file:
           prob_lgb_less_feat = pickle.load(input_file)
           
with open(r"lgb_prob_without_munc.pkl", "rb") as input_file:
           prob_lgb_munc = pickle.load(input_file)
           
with open(r"lgb_prob_calibrated.pkl", "rb") as input_file:
           prob_lgb = pickle.load(input_file)
    
avg_prob=prob_lgb*0.3+prob_lgb_munc*0.2+prob_lgb_ward*0.1+prob_catb_less_feat*0.4

#X_test=np.hstack((prob_lgb,prob_lgb_munc,prob_lgb_ward,prob_catb_less_feat))

#avg_prob1=0.9*avg_prob+0.1*prob_lgb_less_feat

predictions = []

for x in avg_prob:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)

########################voting based
pred_lgb = []

for x in prob_lgb:
    pred_lgb.append(np.argmax(x))
    
pred_lgb_munc = []

for x in prob_lgb_munc:
    pred_lgb_munc.append(np.argmax(x))

pred_lgb_ward = []

for x in prob_lgb_ward:
    pred_lgb_ward.append(np.argmax(x))
    
pred_lgb_catb = []

for x in prob_catb_less_feat:
    pred_lgb_catb.append(np.argmax(x))
    

pred_all=pd.DataFrame({'a1':pred_lgb,'a2':pred_lgb_munc,'a3':pred_lgb_ward,'a4':pred_lgb_catb})

from scipy import stats
pred_all['max_freq'] = stats.mode(pred_all.values, 1)[0]

x=np.array(pred_all['max_freq'])

sub=pd.DataFrame({'building_id':test_df['building_id']})
sub['grade']=x

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)


