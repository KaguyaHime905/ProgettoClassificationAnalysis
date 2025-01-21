INTRODUZIONE 


In questo progetto, è stata condotta un'analisi di Classification Analysis utilizzando un dataset contenente informazioni su 400 clienti di un'azienda. Il dataset includeva variabili descrittive come un ID univoco per ciascun cliente, il genere, l'età e il salario. L'obiettivo principale dell'analisi era prevedere la decisione d'acquisto del cliente, ovvero se ciascun cliente ha deciso di acquistare o meno determinati prodotti offerti dall'azienda.


DESCRIZIONE GENERALE	

Il lavoro si è sviluppato in diverse fasi, a partire dall'esplorazione e dalla pulizia dei dati fino all'applicazione di diversi algoritmi di machine learning per la classificazione. 

    
DESCRIZIONE DEL DATASET: CUSTOMER BEHAVIOUR
            
Il dataset scaricato dal sito Kaggle contiene 400 istanze e 5 colonne. Ecco le varie features:
1.	User ID: Identificatore univoco del cliente (Tipo Attributo: Nominale).
2.	Gender: Genere del cliente (Tipo di Attributo: categorico, Male o Female).
3.	Age: Età del cliente (Tipo Attributo: Ratio).
4.	EstimatedSalary: Salario stimato del cliente (Tipo Attributo: Ratio).
5.	Purchased: Decisione d'acquisto (Tipo di Attributo: Nominale; Binario: 0 = non ha acquistato, 1 = ha acquistato).

Il dataset è completo, non presenta valori mancanti, e può essere utilizzato per applicare modelli di classificazione, con l'obiettivo di prevedere l’etichetta di classe binaria ‘Purchased’. 

PRE-PROCESSING

Pre-processing del Dataset
Per poter applicare in modo corretto gli algoritmi di machine learning, ho eseguito una serie di operazioni di pre-processing sul dataset, al fine di pulirlo e renderlo utilizzabile per i successivi passi di classificazione. L'analisi preliminare del dataset è un passo fondamentale per comprendere la qualità e la struttura dei dati, in modo da identificare eventuali problemi che potrebbero influire sull'analisi successiva.
1.	Caricamento e analisi preliminare del dataset
Ho inizialmente caricato il dataset e analizzato la sua struttura utilizzando alcune funzioni, come .info(), .columns(), e .describe(). Questo mi ha permesso di verificare il numero di righe e colonne, le tipologie di dati, la presenza di eventuali valori mancanti o anomali, e di ottenere una descrizione statistica preliminare dei dati.

2.	Eliminazione della colonna 'User ID'
La colonna 'User ID' è stata eliminata poiché non aggiunge alcuna informazione utile per la classificazione. L'ID è un identificatore univoco del cliente e non influisce sulla decisione d'acquisto. La sua presenza potrebbe anzi introdurre rumore informativo nei modelli di predizione.

3.	Verifica e gestione dei duplicati
Dopo aver rimosso la colonna 'User ID', ho verificato la presenza di duplicati nel dataset. L'analisi ha mostrato che erano presenti delle righe duplicate, che sono state successivamente eliminate per garantire che il dataset fosse pulito.

4.	Riformulazione dei valori della colonna 'Purchased'
I valori nella colonna 'Purchased', che originariamente erano 0 e 1, sono stati sostituiti rispettivamente con "NO" e "YES", per migliorare la leggibilità del dataset durante le fasi esplorative.

5.	Sostituzione dei valori della colonna 'Gender'
La colonna 'Gender', che originariamente conteneva valori categorici ('Male' e 'Female'), è stata trasformata in valori numerici. 'Male' è stato sostituito con 0 e 'Female' con 1. Questa conversione è necessaria per l'algoritmo di classificazione, poiché molti modelli non accettano dati categorici.

6.	Descrizione statistica dei dati rimanenti
Ho posto sotto esame la descrizione della feature, 'EstimatedSalary'. Questo passaggio mi ha fornito informazioni utili, come media, deviazione standard e range dei dati, per una migliore comprensione della loro distribuzione.

Tramite la funzione df.EstimatedSalary.describe(),
ottengo questo output:

Ecco la descrizione dell'attributo EstimatedSalary:
count       380.000000

mean      70421.052632     LA MEDIA
std           34604.155483     DEVIAZIONE STANDARD
min          15000.000000     VALORE MINIMO
25%         43000.000000     1° QUARTILE
50%         70500.000000     2° QUARTILE
75%         88000.000000     3° QUARTILE
max      150000.000000     VALORE MASSIMO


Questo risultato ci verrà utile per l’analisi univariata tramite Box Plot.


ANALISI UNIVARIATA E BIVARIATA CON RELATIVI GRAFICI

Ho deciso inizialmente di rappresentare un Box Plot della feature "EstimatedSalary" per ottenere una visione riassuntiva.

 <img width="482" alt="image" src="https://github.com/user-attachments/assets/d80dea97-be2d-41d4-8169-4fb90377dcb8" />

Descrizione del Grafico:

1.	Il primo quartile (Q1): corrisponde al 25° percentile, il che significa che il 25% dei salari stimati è inferiore a questo valore. Dal grafico, possiamo stimare che Q1 sia a 43.000. Questo significa che un quarto dei customers ha un salario stimato inferiore a 43.000.
2.	Mediana: o secondo quartile (Q2), corrisponde al 50° percentile, il che indica che il 50% dei salari stimati è inferiore a questo valore. La mediana si trova al valore di 70.421. Ciò significa che metà dei customers ha un salario stimato inferiore a 70.421.
3.	Terzo Quartile (Q3): Il terzo quartile corrisponde al 75° percentile, il che significa che il 75% dei salari stimati è inferiore a questo valore. Dal grafico, Q3 corrisponde 88.000, quindi tre quarti dei customers ha un salario stimato inferiore a questa soglia, e solo il 25% guadagna di più.
4.	Intervallo Interquartile (IQR): è la differenza tra il terzo e il primo quartile (Q3 - Q1), e rappresenta il range entro cui si trova il 50% centrale dei dati. In questo caso: 45.000. Questo significa che la maggior parte dei salari stimati varia di circa 45.000 unità tra il primo e il terzo quartile, con la metà dei customer concentrata in questo intervallo.
5.	Le estremità del boxplot indicano l'intervallo di variabilità dei salari senza considerare i valori esterni (outliers). Tali estremità si estendono da 15.000 a 150.000, comprendendo la gran parte dei salari.
6.	Outliers: Non sembrano esserci valori anomali nel grafico, poiché non si vedono punti isolati al di fuori delle estremità. 

Riassumendo:
•	Q1 = 43.000 (25% dei dati sono inferiori a questo valore)
•	Mediana (Q2) = 70.421 (50% dei dati sono inferiori a questo valore)
•	Q3 = 88.000 (75% dei dati sono inferiori a questo valore)
•	IQR = 45.000 (Range Interquartile)


ANALISI BIVARIATA

GENDER - ESTIMATED SALARY

<img width="482" alt="image" src="https://github.com/user-attachments/assets/1561a713-4228-4ae6-9149-9daf0855ff96" />

OUTPUT:
Valori dei quartili per ogni categoria di Gender:
Gender   Q1          Mediana       Q3
0              43500.0  71000.0      87000.0		(MALE)
1              43250.0  68500.0      94500.0		(FEMALE)

Analisi del Grafico

Il boxplot mostra la distribuzione di EstimatedSalary in base al Gender. 
A partire dai risultati ottenuti in output vediamo come entrambi i gender, Male e Female presentano più o meno l medesimo Q1; quindi, il 25% di utenti maschili e femminili comunque ha uno stipendio che parte orientativamente da sotto i 20.000 fino ai 43.250/43.550. 
Anche a livello di Mediana la differenza è di 2.500. Quindi il 50% degli utenti maschili ha stipendi inferiori a 71.000 mentre il 50% di utenti femminili ha stipendi inferiori a 68.500.
Per quanto riguarda Q3, invece in questo caso specifico il valore per gli utenti femminili di EstimatedSalary arriva a 94.500 mentre gli utenti maschili arriva a 87.000, quindi possiamo affermare che il 75% di utenti femminili ha uno stipendio inferiore a 94.500 invece il 75% di utenti maschili presenta uno stipendio inferiore a 87.000. 
A prescindere però dai puri dati statistici, è anche vero che se consideriamo la distribuzione degli individui per ogni quartile, è visibile che, seppur minima, esiste comunque un divario fra gli stipendi percepiti degli utenti maschili e gli utenti femminili. Il divario a cui faccio riferimento è un divario QUANTITATIVO, ossia di distribuzione entro ogni quartile (considerando che il 52% degli utenti sono femminili e il restante 48% sono utenti maschili.)


ANALISI BIVARIATA

ESTIMATED SALARY- PURCHASED

<img width="482" alt="image" src="https://github.com/user-attachments/assets/d7275862-012c-4502-8ca2-a958dd4af001" />

OUTPUT:
Valori dei quartili per ogni categoria di Purchased:
Purchased  Q1           Mediana   Q3
NO         	     43750.0  61000.0   78250.0
YES              41750.0  92000.0  123500.0

Analisi del Grafico

Il boxplot mostra la distribuzione di EstimatedSalary in base al valore assunto dall’attributo Purchased (YES/NO). 
Ancora prima di osservare i dati ottenuti in output, è evidente la presenza di una correlazione fra i valori di EstimatedSalary con l’esito positivo dell’acquisto, è quindi possibile ipotizzare che il prodotto di acquisto sia più accessibile per individui con un salario più elevato. In questa analisi il fulcro è analizzare le caratteristiche dell’utente che decide di procedere all’acquisto:
osservando la mediana che corrisponde al valore 92.000 di EstimatedSalary, si evince come il 50% di utenti con uno stipendio inferiore a 92.000 decida comunque di procedere all’acquisto del prodotto. Altrettanto interessante è osservare il valore di Q3 corrispondente al valore di 123.500, ossia il 75% di utenti che decide di procedere all’acquisto presenta un salario pari o inferiore a tale cifra. La cifra massima rappresentata dagli utenti che decidono di procedere all’acquisto supera addirittura i 140.000. 
Altro accorgimento importante, sempre dal punto di vista QUANTITATIVO: si noti la concentrazione del 25% di utenti il cui stipendio oscilla fra i 92.000 e i 123.000 e l’altro 25% il cui stipendio oscilla fra i 41.750 e i 92.000 estremamente diluito in questo intervallo.

SCATTERPLOT ESTIMATED SALARY – AGE

 <img width="482" alt="image" src="https://github.com/user-attachments/assets/d0f82a4f-5af0-4efe-892d-d38a98d1eb1d" />

Analisi del Grafico
Lo  scatterplot mostra la relazione tra EstimatedSalary e l’età (Age). I punti sono distribuiti in modo abbastanza disperso su tutto il grafico, senza presentare una correlazione lineare (positiva o negativa) tra l’età e lo stipendio. Da questa rappresentazione quindi l’età non sembra influenzare lo stipendio.
Interessante è notare che gli stipendi sono variabili per tutte le età, indicando che ci sono persone con stipendi molto diversi all’interno di ogni fascia di età.

GRAFICO TORTA PERCENTUALE UOMO-DONNA

<img width="482" alt="image" src="https://github.com/user-attachments/assets/66e350fe-112f-41e1-9413-3b4581d54683" />
 
GRAFICO TORTA PURCHASED

<img width="482" alt="image" src="https://github.com/user-attachments/assets/ee4c0b15-6f1f-4484-8835-bd8b502f3f37" />


Col comando autopct='%1.1f%%' ottengo la percentuale con una cifra decimale.


ALGORITMO DI CLASSIFICAZIONE CODICE:

 <img width="482" alt="image" src="https://github.com/user-attachments/assets/a9b57e7f-2ca7-4e82-9fd5-44f482fbac51" />
 


Decision Tree
Per la fase di classificazione di questo progetto ho utilizzato un Decision Tree Classifier (Albero di Decisione).


ALGORITMO DI CLASSIFICAZIONE

Preparazione dei dati
Prima di addestrare il modello il dataset è stato suddiviso nelle seguenti parti:
1.	Attributi descrittivi (X):
o	Gli attributi che descrivono i clienti sono: Gender (codificato come 0 e 1), Age e EstimatedSalary. 
2.	Attributi di classe (y):
o	L’etichetta di classe da prevedere è Purchased (NO/YES), che indica se un cliente ha acquistato o meno il prodotto. La colonna è stata convertita in stringa per agevolare la classificazione.

Suddivisione dei dati in training e test set
Al fine dell’addestramento del modello, il dataset iniziale è stato suddiviso in: 
•	Training set (70%): Utilizzato per addestrare il modello.
•	Test set (30%): Utilizzato per testare le performance del modello su dati ‘sconosciuti’, possiamo definirli nuovi.
La suddivisione viene effettuata tramite la funzione train_test_split(), impostando una dimensione del test set del 30% e un random_state=0 (chiamato anche ‘seme’) per rendere nuovamente riproducibili i risultati.

Ricerca della profondità ottimale dell'albero
Uno dei parametri più importanti per un albero decisionale è la sua profondità (max_depth), che determina il livello massimo di suddivisione dei nodi. Un albero troppo profondo può risultare ‘overfitted’, ossia troppo specializzato nel riprodurre i dati di addestramento, mentre un albero troppo poco profondo potrebbe essere ‘underfitted’, ossia incapace di catturare relazioni complesse nei dati.
Sulla base di queste considerazioni ho effettuato una ricerca della profondità ottimale eseguendo un ciclo su diversi livelli di profondità (da 1 a 8) tramite il codice livelli=range(1,9). 
Per ciascun livello, ho utilizzato la tecnica di cross-validation.
Da questa analisi, la profondità ottimale risultante è stata 5, per ottenere un modello bilanciato. 

Addestramento e valutazione del modello
Una volta determinata la profondità ottimale, ho addestrato il modello di Decision Tree con la profondità impostata a 5 e con un seme = 1 sempre per permettere la riproducibilità del risultato. 

Tramite questo codice:

dtc_model=DecisionTreeClassifier(max_depth=5, random_state=1)
dtc_model.fit(X_train,y_train)
etichette_predette=dtc_model.predict(X_test)
etichette_predette=pd.DataFrame(etichette_predette,columns=['Etichette Predette'])
accuratezza=accuracy_score(y_test,etichette_predette)
print("L'accuratezza di tale previsione è:", accuratezza)

X_train: Contiene le variabili indipendenti.
Age, EstimatedSalary, Gender (precedentemente salvate nella variabile attributi_descrittivi e poi resi un sub-set (X).
y_train: Contiene le etichette.
I valori che assume l’attributo Purchased (YES / NO).
Una volta ottenuto il modello di apprendimento, questo viene applicato agli attributi descrittivi che compongono il sub-set di Test, tramite questa linea di codice: etichette_predette=dtc_model.predict(X_test)
E si crea un data frame contenente una colonna denominata in questo caso columns=['Etichette Predette'].
Si calcola infine l’accuratezza, ossia si fa un confronto fra le etichette predette dal modello addestrato e le ETICHETTE ORIGINALI DEL DATASET INIZIAZLE (della cui accuratezza siamo certi al 100%): accuratezza=accuracy_score(y_test,etichette_predette)

Nel caso specifico è risultata essere 89.47%. 

L'accuratezza indica la percentuale di elementi nel test set che sono stati classificati correttamente. Un'accuratezza del 89.47% indica che il modello riesce a prevedere correttamente il comportamento d'acquisto della maggior parte dei clienti.


RAPPRESENTAZIONE TESTUALE DELL’ALBERO
Albero Decisionale in rappresentazione testuale: 

|--- Age <= 41.5000 			DIRAMAZIONE A SINISTRA
| |--- EstimatedSalary <= 91500.0000 
| | |--- Age <= 36.5000 
| | | |--- class: NO 
| | |--- Age > 36.5000 
| | | |--- EstimatedSalary <= 83500.0000 
| | | | |--- Age <= 37.5000 
| | | | | |--- class: NO 
| | | | |--- Age > 37.5000 
| | | | | |--- class: NO 
| | | |--- EstimatedSalary > 83500.0000 
| | | | |--- class: YES 
| |--- EstimatedSalary > 91500.0000 
| | |--- Age <= 26.5000 
| | | |--- class: NO 
| | |--- Age > 26.5000 
| | | |--- EstimatedSalary <= 116000.0000 
| | | | |--- EstimatedSalary <= 107500.0000 
| | | | | |--- class: YES 
| | | | |--- EstimatedSalary > 107500.0000 
| | | | | |--- class: NO 
| | | |--- EstimatedSalary > 116000.0000 
| | | | |--- class: YES 
|--- Age > 41.5000 			DIRAMAZIONE A DESTRA
| |--- Age <= 46.5000 
| | |--- EstimatedSalary <= 85000.0000 
| | | |--- EstimatedSalary <= 52000.0000 
| | | | |--- Age <= 44.5000 
| | | | | |--- class: NO 
| | | | |--- Age > 44.5000 
| | | | | |--- class: YES 
| | | |--- EstimatedSalary > 52000.0000 
| | | | |--- EstimatedSalary <= 73500.0000 
| | | | | |--- class: NO 
| | | | |--- EstimatedSalary > 73500.0000 
| | | | | |--- class: NO 
| | |--- EstimatedSalary > 85000.0000 
| | | |--- Gender <= 0.5000 	 UTENTE MASCHIO
| | | | |--- class: YES 
| | | |--- Gender > 0.5000 	UTENTE FEMMINA
| | | | |--- Age <= 45.5000 
| | | | | |--- class: YES 
| | | | |--- Age > 45.5000 
| | | | | |--- class: NO 
| |--- Age > 46.5000 
| | |--- EstimatedSalary <= 40500.0000 
| | | |--- class: YES 
| | |--- EstimatedSalary > 40500.0000 
| | | |--- EstimatedSalary <= 45000.0000 
| | | | |--- Age <= 59.5000 
| | | | | |--- class: NO 
| | | | |--- Age > 59.5000 
| | | | | |--- class: YES 
| | | |--- EstimatedSalary > 45000.0000 
| | | | |--- Age <= 52.5000 
| | | | | |--- class: YES 
| | | | |--- Age > 52.5000 
| | | | | |--- class: YES

ALBERO DECISIONALE SPIEGAZIONE
La motivazione che mi ha spinto ad allegare questo formato testuale rispetto all’immagine è dovuta al fatto che non sono riuscita a ottenere un’immagine leggibile dell’albero in quanto anche cambiando il formato di immagine il risultato era il medesimo. Però è anche vero che questo formato testuale ci permette di seguire i passaggi effettuati dal modello di apprendimento.

L'albero decisionale ottenuto rappresenta il processo di classificazione del comportamento d'acquisto dei clienti in base a tre attributi: Age, EstimatedSalary e Gender. 
Ad ogni livello dell'albero, il dataset viene suddiviso in base ai valori di queste variabili, con il risultato di etichetta di classe che rappresenta se un cliente ha deciso di acquistare (class: YES) o meno (class: NO) un prodotto.

Struttura dell'albero decisionale
•	Nodo radice: La prima suddivisione avviene in base all'età del cliente:
o	Se l'età è inferiore o uguale a 41.5 anni, l'albero si dirama a sinistra.
o	Se l'età è superiore a 41.5 anni, l'albero si dirama 
a destra.

Percorso per clienti di età ≤ 41.5 anni
1.	EstimatedSalary ≤ 91.500: Viene effettuata una nuova suddivisione in base all'età.
o	Se l'età è inferiore o uguale a 36.5 anni, il modello prevede che il cliente non acquisterà il prodotto (class: NO).
o	Se l'età è superiore a 36.5 anni, l'algoritmo si concentra sul salario stimato:
o	Se il salario è inferiore a 83.500, viene considerata nuovamente l'età, ma tutti i rami portano alla previsione di non acquisto.
o	Se il salario supera 83.500, il cliente acquisterà il prodotto (class: YES).
2.	EstimatedSalary > 91.500: A questo punto, viene effettuata una nuova suddivisione basata sull'età:
o	Se l'età è inferiore o uguale a 26.5 anni, il cliente non acquisterà (class: NO).
o	Se l'età è maggiore di 26.5 anni, la previsione si basa sul salario stimato:
o	Se il salario è inferiore a 116.000, vengono fatte ulteriori verifiche sul salario, portando a predizioni sia di acquisto per valori di salario inferiore a 107.500, che di non acquisto per valori di salario superiori a 107.500.
o	Se il salario è superiore a 116.000, il cliente acquisterà (class: YES).

Percorso per clienti di età > 41.5 anni
1.	Age ≤ 46.5 anni:
o	Se il salario stimato è inferiore a 52.000, l'albero si concentra ancora sull'età, con predizioni sia di acquisto che di non acquisto a seconda della fascia di età, più precisamente per utenti di età inferiore a 44 anni, l’etichetta predetta è NO; per utenti con età superiore a 44 anni, l’etichetta predetta è YES.
o	Se il salario è maggiore, il genere del cliente gioca un ruolo:
o	Se il cliente è maschio (Gender = 0), acquista (class: YES).
o	Se è femmina (Gender = 1), l'albero continua a suddividere per stipendi, portando a previsioni sia di acquisto per un EstimatedSalary <= 45.500, che di non acquisto per valori di EstimatedSalary > 45.500.
2.	Age > 46.5 anni:
o	Se il salario è inferiore a 40.500, il cliente acquisterà (class: YES).
o	Se il salario è maggiore, l'albero prende in considerazione ulteriori suddivisioni basate su età e salario, con predizioni sia di acquisto che di non acquisto.

Conclusioni
L'albero decisionale mostra come l'algoritmo utilizza le informazioni sull'età, salario (sono i fattori più influenti) e genere (preso molto meno in considerazione) per prendere decisioni sul comportamento d'acquisto. 








