import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# 1
market = pd.read_csv(r'market_basket.txt', delimiter="\t")
m = market.describe()
print("description du market :")
print(m)

# 2
print("2. les 10 premières lignes du DataFrame :")
result = market.head(10)
print(result)

# 3
print("3. les dimensions du dataframe :")
print(market.shape)


# 4
def construire():
    crosstable = pd.DataFrame(index=range(int(market["ID"].iloc[-1:])))
    crosstable["ID"] = range(1,int(market["ID"].iloc[-1:]+1))
    for i in market.index:
        prod = market["Product"][i]
        crosstable[prod] = 0
        for j in crosstable.index:
            if (market["ID"][i] == crosstable["ID"][j]):
                crosstable[prod][j] = 1
    return (crosstable)
print("fonction pour construire un tableau binaire")
print(construire().head(5))

#5
print("bibliotheque Crosstab : ")
cross = pd.crosstab(market['ID'], market['Product'])
print(cross)


#6
print("Afficher les 30 premières transactions et les 3 premiers produits :")
print(cross.iloc[:30,:3])


#7
print("fonction apriori :")
frequent_itemsets = apriori(cross, min_support=0.025, max_len=4, use_colnames=True)
#frequent_itemsets = fpgrowth(cross, min_support=0.025, max_len=4, use_colnames=True)

print(frequent_itemsets.shape)
#8
print("Afficher les 15 premiers itemsets : ")
print(frequent_itemsets.head(15))

#9
print("fonction is_inclus() : ")
def is_inclus(x,items):
    return items.issubset(x)

#10
print("Afficher les itemsets comprenant le produit ‘Aspirin’:")
#print(frequent_itemsets[frequent_itemsets['itemsets'].ge({'Aspirin'})])
id = np.where(frequent_itemsets.itemsets.apply(is_inclus,items={'Aspirin'}))
print(frequent_itemsets.loc[id])



#11
print("Afficher les itemsets contenant Aspirin et Eggs : ")
#print(frequent_itemsets[frequent_itemsets['itemsets'].ge({'Aspirin','Eggs'})])
id2 = np.where(frequent_itemsets.itemsets.apply(is_inclus,items={'Aspirin','Eggs'}))
print(frequent_itemsets.loc[id2])

#12
print("minimal (min_threshold = 0.75) sur une mesure d’intérêt, en l’occurrence la confiance dans notre exemple (metric = ‘’confidence’’)")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75)
print(rules.columns)

#13
print("Afficher les 5 premières règles : ")
print(rules.iloc[:5,:])

#14
print("Filtrer les règles en affichant celles qui présentent un LIFT supérieur ou égal à 7.")
print(rules[rules['lift'].ge(7.0)])


#15
print("Filtrer les règles en affichant celles menant au conséquent {‘2pct_milk’} :")
print(rules[rules['consequents'].eq({'2pct_Milk'})])

