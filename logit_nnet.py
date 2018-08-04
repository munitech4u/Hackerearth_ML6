comb=pd.read_csv("comb.csv")

flots=[col for col in comb.columns if comb[col].dtype == 'float64']
ints=[col for col in comb.columns if comb[col].dtype == 'int64']

for i in range(len(flots)):
    comb[flots[i]]=comb[flots[i]].astype('float32')
    
for i in range(len(ints)):
    comb[ints[i]]=comb[ints[i]].astype('int32')
    
cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','has_repair_started']


#cats = [col for col in comb.columns if hasattr(comb[col],'cat') is True and col not in['building_id','damage_grade','vdcmun_id','ward_id','index']]

comb1 = pd.get_dummies(comb, columns= cats, dummy_na=True)
train_df = comb1.iloc[:i_len].fillna(0)
test_df = comb1.iloc[i_len:].fillna(0)

v4=list(set(train_df.columns)-set(['vdcmun_id','ward_id','building_id','damage_grade','index']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()



from sklearn.linear_model import LogisticRegressionCV

logit=LogisticRegressionCV(multi_class='ovr')

logit.fit(train_df[v4],label)
prob_logit=logit.predict_proba(test_df[v4])





from sklearn.neural_network import MLPClassifier
nnet=MLPClassifier(hidden_layer_sizes=(20,20,20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.05, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001, verbose=True)

nnet.fit(train_df[v4],label)
prob_nnet=nnet.predict_proba(test_df[v4])

predictions = []

for x in prob_nnet:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)
