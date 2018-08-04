# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 19:09:47 2018

@author: vvootl010
"""

cats = [col for col in comb.columns if hasattr(comb[col],'cat') is True and col not in['building_id','damage_grade','district_id','vdcmun_id','ward_id']]

comb1 = pd.get_dummies(comb, columns= cats, dummy_na=True)

v4=list(set(comb1.columns)-set(['index','district_id','vdcmun_id','ward_id','g1','g2','g3','g4','g5','label','damage_grade','building_id']))


from sklearn import preprocessing
X_scaled = preprocessing.scale(comb1[v4].fillna(0))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=20)
kmeans.fit(X_scaled)

labels = kmeans.predict(X_scaled)

testing=pd.DataFrame({'target':train_df1['label'],'clus':labels})

x=testing.groupby(['clus','target']).size().reset_index()


comb['clus']=labels