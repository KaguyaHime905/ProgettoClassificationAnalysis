import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
import warnings
# noqa: F401,F403
warnings.filterwarnings('ignore')

## INIZIO ANALISI E PREPROCESSING ##
dataset=pd.read_csv('Customer_Behaviour.csv')
print("Ecco le info del dataset analizzato:\n",dataset.info)
print("Ecco i nomi delle colonne del dataset analizzato:\n", dataset.columns)
print("Ecco la descrizione del dataset analizzato:\n", dataset.describe())
#Elimino la colonna User ID perchè non posso basare la classificazione sugli ID degli utenti
new_dataset=dataset.drop('User ID', axis=1, inplace=False)
print("Ecco le colonne nel nuovo dataset:\n",new_dataset.columns)
print(new_dataset.shape)
print("Ecco la descrizione del nuovo dataset analizzato senza la colonna User ID:\n", new_dataset.describe())
print(new_dataset.isnull().sum())
print("Di seguito i valori duplicati del dataset caricato:")
print(new_dataset[new_dataset.duplicated])
## PROCEDO ELIMINANDO I DUPLICATI ##
new_dataset.drop_duplicates(inplace=True)
print(new_dataset.shape)

df=pd.DataFrame(new_dataset)
#SOSTITUISCO I VALORI 0 E 1 CON YES E NO PER UNA LETTURA PIU' COMPRENSIBILE
df['Purchased'] = df['Purchased'].replace({0: 'NO', 1: 'YES'})
print(df.info)
print("Ecco la descrizone dell'attributo EstimatedSlary:\n", df.EstimatedSalary.describe())
#SOSTITUISCO I VALORI MALE E FEMALE CON 0 CON 1 PER LA LETTURA DA PARTE DELL'ALGORITMO DI CLASSIFICAZIONE
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female':1})
print("Ecco la descrizone dell'attributo Gender:\n", df.Gender.describe())

#ANALISI UNIVARIATA:
#BOXPLOT PER SALARY
plt.figure(figsize=(10,6))
sns.boxplot(x='EstimatedSalary', data=df)
plt.title('Distribuzione EstimatedSalary', size=18)
plt. show()

##ANALISI BIVARIATA:
#BOXPLOT FRA ATTRIBUTI NUMERI E CATEGORICI
#Boxplot EstimatedSalary - Gender
plt.figure(figsize=(10,6))
sns.boxplot(x='Gender', y='EstimatedSalary', data=df)
plt.title('Distribuzione EstimatedSalary - Gender', size=18)
quartili = df.groupby('Gender')['EstimatedSalary'].quantile([0.25, 0.5, 0.75]).unstack()
quartili.columns = ['Q1', 'Mediana', 'Q3']
print("Valori dei quartili per ogni categoria di Gender:", quartili)
plt. show()

#Boxplot EstimatedSalary - Purchased
plt.figure(figsize=(10,6))
sns.boxplot(x='Purchased', y='EstimatedSalary', data=df)
plt.title('Distribuzione EstimatedSalary - Purcased', size=18)
quartili = df.groupby('Purchased')['EstimatedSalary'].quantile([0.25, 0.5, 0.75]).unstack()
quartili.columns = ['Q1', 'Mediana', 'Q3']
print("Valori dei quartili per ogni categoria di Purchased:", quartili)
plt. show()     #ci sono degli outliers

#Grafici a torta
plt. figure(figsize=(10,6))
df.Gender.value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
plt.title('Percentuale di Uomini - Donne')
plt.show()

plt.figure(figsize=(10,6))
df.Purchased.value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
plt.title('Percentuale valori di Purchased')
plt.show()

#Scatterplot fra AGE - SALARY
plt.figure(figsize=(10,6))
sns.scatterplot(x='EstimatedSalary', y='Age', data=df)
plt.title('Relazione fra EstimatedSalary - Age', size=18)
plt.show()


##CLASSIFICAZIONE
attributi_decrittivi=['Gender','Age','EstimatedSalary']
X=df[attributi_decrittivi]
attributi_classe=['Purchased']
y=df[attributi_classe]

y[attributi_classe]=y[attributi_classe].astype(str)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)

livelli = range(1, 9)
valori = []

for livello in livelli:
    model = DecisionTreeClassifier(max_depth=livello)
    score = cross_val_score(model, X_train, y_train, cv=5)
    valori.append(score.mean())

livello_ottimale = livelli[valori.index(max(valori))]
print("La profondità ottimale dell'albero è:", livello_ottimale)        #La profondità ottimale è: 5

dtc_model=DecisionTreeClassifier(max_depth=5, random_state=1)
dtc_model.fit(X_train,y_train)
etichette_predette=dtc_model.predict(X_test)
etichette_predette=pd.DataFrame(etichette_predette,columns=['Etichette Predette'])

accuratezza=accuracy_score(y_test,etichette_predette)
print("L'accuratezza di tale previsione è:", accuratezza)   #L'accuratezza di tale previsione è: 0.8947368421052632

rappresentazione_testuale= tree.export_text(dtc_model, feature_names=attributi_decrittivi, decimals=4)
print("Albero Decisionale in rappresentazione testuale:\n",rappresentazione_testuale)
with open("classification_tree_Customer Behaviour.txt", "w") as fout:
    fout.write(rappresentazione_testuale)
print("Nomi degli attributi descrittivi:", attributi_decrittivi)
print("Valori di classe:", dtc_model.classes_)

fig=plt.figure(figsize=(11,6))
grafico_albero_decisionale=tree.plot_tree(dtc_model,feature_names=attributi_decrittivi,class_names=dtc_model.classes_, filled=True)
plt.show()
fig.savefig("Albero di Classificazione Customer Behaviour.png")
fig.savefig("Albero di Classificazione Customer Behaviour.pdf")