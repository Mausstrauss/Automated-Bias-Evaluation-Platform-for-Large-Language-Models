# **Analyse und Operationalisierung von Bias-Kennzahlen und Messmethoden für Large Language Models**

## **TEIL I: FUNDAMENTALE KONZEPTE UND TAXONOMIE DER BIAS-MESSUNG**

### **1\. Einleitung: Die Notwendigkeit einer kontinuierlichen, automatisierten Bias-Evaluation**

Large Language Models (LLMs), wie sie von führenden KI-Laboren (z. B. OpenAI, Deepseek, X-AI) entwickelt werden, durchdringen zunehmend reale Anwendungen. Ihre schnelle Evolutionsrate – gekennzeichnet durch häufige Modellaktualisierungen, neue Versionen und diverse Deployment-Konfigurationen – stellt die Forschung vor erhebliche Herausforderungen. Während Bias in spezifischen Modellen zu bestimmten Zeitpunkten analysiert wurde, veralten diese statischen Analysen schnell. Dies offenbart eine kritische Lücke: Es fehlt an einer systematischen, kontinuierlichen Evaluierungsmethodik, die es Forschern und Praktikern ermöglicht, die Evolution von Bias über Modellgenerationen hinweg zu verfolgen.  
Das Ziel dieses Berichts ist die Schaffung einer umfassenden wissenschaftlichen und methodischen Grundlage für eine automatisierte Evaluierungsplattform. Dies erfordert eine rigorose Extraktion, Systematisierung und Analyse von Kennzahlen, mathematischen Methoden und Analyseverfahren, wie sie im aktuellen Stand der Forschung dokumentiert sind.  
Eine zentrale Herausforderung für eine solche Plattform ist die Variabilität im Modellzugriff. Die Evaluierung muss daher modular konzipiert sein, um fundamental unterschiedliche Szenarien zu behandeln:

1. **Black-Box-Evaluierung:** Gilt für proprietäre Modelle, die nur über APIs zugänglich sind (z. B. OpenAI). Hier ist die Analyse auf die Auswertung von Textausgaben (Generierter Text) beschränkt.  
2. **White-/Grey-Box-Evaluierung:** Gilt für quelloffene Modelle (z. B. Llama-Varianten). Hier besteht voller Zugriff auf interne Modellzustände, einschließlich Einbettungen (Embeddings) und Wahrscheinlichkeitsverteilungen (Logits), was tiefere diagnostische Analysen erlaubt.

### **2\. Formale Definitionen: Was wird gemessen?**

Um Bias messbar zu machen, ist eine präzise Begriffsbestimmung unerlässlich.  
**Bias vs. Fairness** In diesem Kontext wird Bias als "systematische und unfaire Diskriminierung gegen bestimmte Individuen oder Gruppen" oder formaler als "disparate Behandlung oder Ergebnisse zwischen sozialen Gruppen, die aus historischen und strukturellen Machtasymmetrien entstehen" definiert. Fairness ist das normative Ziel, diese unerwünschten Verzerrungen zu minimieren.  
**Taxonomie der Schäden (Harms)** Die durch Bias verursachten Schäden lassen sich in zwei Hauptkategorien unterteilen :

1. **Allokative Schäden (Allocational Harms):** Beziehen sich auf die ungerechte Verteilung von Ressourcen oder Möglichkeiten. Dies ist typisch für nachgelagerte Klassifikationsaufgaben (z. B. Kreditwürdigkeitsprüfung).  
2. **Repräsentationale Schäden (Representational Harms):** Beziehen sich auf die Art und Weise, wie soziale Gruppen dargestellt werden. Dies ist der primäre Fokus bei der Evaluierung generativer LLMs. Diese Kategorie umfasst:  
   * **Stereotypisierung (Stereotyping):** Assoziation von Gruppen mit stereotypen Attributen.  
   * **Demütigung (Denigration/Toxicity):** Erzeugung von toxischen, hasserfüllten oder beleidigenden Inhalten in Bezug auf eine Gruppe.  
   * **Mangelnde Repräsentation (Exclusionary Norms):** Das Versäumnis, bestimmte Gruppen anzuerkennen oder adäquat darzustellen.

**Taxonomie der Bias-Dimensionen** Eine multi-dimensionale Evaluierungsplattform muss Bias entlang verschiedener "Achsen" messen. Die Forschung hat einen Kanon von Bias-Typen etabliert, die als Testsuiten dienen :

* Geschlecht (Gender)  
* Ethnie/Herkunft (Race)  
* Religion  
* Alter (Age)  
* Nationalität (Nationality)  
* Sexuelle Orientierung (Sexual Orientation)  
* Behinderung (Disability)  
* Äußeres Erscheinungsbild (Physical Appearance)  
* Sozioökonomischer Status (Socioeconomic Status)

Es ist jedoch von entscheidender Bedeutung zu erkennen, dass diese allgemeinen Kategorien für spezialisierte Anwendungsdomänen unzureichend sein können. Eine robuste Plattform muss erweiterbar sein, um domänenspezifische Bias-Profile zu unterstützen, wie z. B. **Health Equity Bias** im medizinischen Bereich. Dieser spezifische Bias-Typ umfasst die "Nichtberücksichtigung struktureller Erklärungen für Ungleichheiten" oder das "unverhältnismäßige Vorenthalten von Möglichkeiten" basierend auf demografischen Merkmalen in medizinischen Ratschlägen.wie beispielsweise der **Health Equity Bias** im Gesundheitswesen. Bei dieser speziellen Art von Bias geht es darum, dass man "strukturelle Erklärungen für Ungleichheiten nicht berücksichtigt" oder "Chancen unverhältnismäßig vorenthält" – basierend auf demografischen Merkmalen in medizinischen Empfehlungen.

### **3\. Eine methodische Taxonomie der Bias-Messverfahren**

Die in der Forschung identifizierten Methoden zur Quantifizierung von Bias lassen sich, wie in Abschnitt 1 dargelegt, am sinnvollsten nach der erforderlichen Zugriffsebene auf das Modell klassifizieren.  
**Ansatz 1: Embedding-basierte Metriken (White-Box)** Diese Methoden analysieren die internen Modellrepräsentationen (Einbettungen) und operieren auf der Annahme, dass semantische Assoziationen durch die geometrische Nähe im Vektorraum repräsentiert werden.

* **Prinzip:** Messung der Assoziationsstärke zwischen Vektoren für soziale Gruppen (z. B. "Mann", "Frau") und Vektoren für stereotype Konzepte (z. B. "Karriere", "Familie").  
* **Erfordert:** Vollen Zugriff auf die Wort- oder Satz-Embedding-Matrizen des Modells.  
* **Beispiele:** Word Embedding Association Test (WEAT), Sentence Encoder Association Test (SEAT), Kategoriale Assoziation (Cramér's V).

**Ansatz 2: Wahrscheinlichkeitsbasierte Metriken (Grey-Box)** Diese Methoden analysieren die Wahrscheinlichkeitsverteilung (Logits) des Modells für das nächste Token. Sie werden oft als "Grey-Box" bezeichnet, da sie keinen Zugriff auf die vollen Gewichte, aber mehr als nur die finale Ausgabe benötigen.

* **Prinzip:** Vergleich der Wahrscheinlichkeit (typischerweise Pseudo-Log-Likelihood), die das Modell stereotypen Sätzen im Vergleich zu anti-stereotypen Sätzen zuweist.  
* **Erfordert:** Zugriff auf die Logit-Ausgaben des Modells.  
* **Beispiele:** CrowS-Pairs Score, Context Association Test (CAT), All Unmasked Likelihood (AUL), Log-Probability Bias Score (LPBS).

**Ansatz 3: Generierte Text-basierte Metriken (Black-Box)** Diese Methoden analysieren ausschließlich die finale Textausgabe des Modells als Reaktion auf einen Prompt. Dies ist der universellste Ansatz und der einzige, der für alle kommerziellen API-basierten Modelle anwendbar ist.

* **Prinzip:** Bewertung der Qualität, Toxizität, des Sentiments oder der stereotypen Natur des generierten Textes.  
* **Erfordert:** Lediglich API-Zugriff (Text-Input, Text-Output).  
* **Beispiele:** Klassifikator-basierte Metriken (Toxizität, Sentiment), Verteilungsmetriken (Wasserstein-Distanz).

Eine fundamentale Erkenntnis aus der aktuellen Forschung ist die **schwache oder inkonsistente Korrelation** zwischen intrinsischen Metriken (Ansatz 1 & 2\) und extrinsischen, realen Schäden (Ansatz 3). Modelle, die durch Reinforcement Learning from Human Feedback (RLHF) trainiert wurden, können intrinsische Tests wie CrowS-Pairs oft gut "bestehen", indem sie expliziten Bias maskieren, während impliziter oder generativer Bias (im Black-Box-Szenario) weiterhin vorhanden ist. Für die angestrebte Plattform bedeutet dies, dass ein "Generation-First"-Ansatz (Black-Box) priorisiert werden muss, um aussagekräftige und robuste Ergebnisse zu liefern.

## **TEIL II: MATHEMATISCHE METHODEN UND KENNZAHLEN IM DETAIL**

Dieser Teil extrahiert die spezifische "Messtechnik bzw. Mathematik", die zur Berechnung der Bias-Kennzahlen erforderlich ist.

### **4\. White-Box: Metriken auf Einbettungsebene**

Diese Metriken sind nur auf Open-Source-Modelle anwendbar, bei denen die Einbettungsvektoren extrahiert werden können.

#### **4.1. Word/Sentence Embedding Association Test (WEAT/SEAT)**

**Konzept:** WEAT (und seine Satz-basierte Variante SEAT) misst die differentielle Assoziation von zwei Sätzen von Zielkonzepten (z. B. A \= Berufe, B \= Haushaltsgegenstände) mit zwei Sätzen von Attributen (z. B. X \= männliche Namen, Y \= weibliche Namen).  
**Mathematische Grundlage:** Die Berechnung folgt :

1. Definiere eine Ähnlichkeitsfunktion s(w, A, B) für ein Wort w und die Konzeptmengen A und B: wobei \\cos(w, a) die Kosinus-Ähnlichkeit zwischen den Embedding-Vektoren von w und a ist.  
2. Definiere die Teststatistik s(X, Y, A, B), welche die Summe der differentiellen Assoziationen für alle Attributwörter in X und Y vergleicht:  
3. Die finale **Kennzahl ist die Effektstärke d**, die den Unterschied der Mittelwerte durch die Standardabweichung normalisiert: Ein Wert d \> 0 indiziert eine stärkere Assoziation von X mit A (und Y mit B), während d \< 0 das Gegenteil anzeigt. Ein Wert nahe 0 signalisiert Neutralität.

#### **4.2. Kategoriale Assoziation (Cramér's V)**

**Konzept:** Diese in vorgestellte Methode nutzt einen Hilfs-Klassifikator, um statistische Korrelationen zwischen dem Raum der geschützten Attribute und dem Raum der stereotypen Attribute im Embedding-Space zu quantifizieren.  
**Mathematische Grundlage:** Der Prozess ist mehrstufig :

1. **Training:** Es wird ein einfacher linearer Klassifikator (z. B. Linear Support Vector Machine, LSVM) trainiert. Der Trainingsdatensatz besteht aus den Embeddings von Wörtern, die *geschützte Attribute* klar definieren

 (z. B. [![][image1]](https://www.codecogs.com/eqnedit.php?latex=X_%7B%5Ctext%7Btrain%7D%7D#0) \= [![][image2]](https://www.codecogs.com/eqnedit.php?latex=%5C%7B%5Ctext%7BEmbeddings%20f%C3%BCr%20'christlich'%2C%20'muslimisch'%2C%20'j%C3%BCdisch'%2C...%7D%5C%7D\).#0)

2. **Inferenz:** Der trainierte Klassifikator wird nun verwendet, um die Embeddings von Wörtern vorherzusagen, die *stereotype Attribute* beschreiben (z. B.   
3. ![][image3]  
4. **Messung:** Die Ergebnisse werden in einer Kontingenzmatrix C erfasst, welche die tatsächlichen stereotypen Klassen (z. B. positive, negative) gegen die vom Klassifikator vorhergesagten geschützten Klassen (z. B. christlich, muslimisch) stellt.  
5. Für jede Zelle (p, s) in der Matrix (wobei p die vorhergesagte geschützte Klasse und s die tatsächliche stereotype Klasse ist) wird die beobachtete Häufigkeit Of(p, s) mit der erwarteten Häufigkeit Ef(p, s) verglichen. Die erwartete Häufigkeit unter der Annahme der Unabhängigkeit ist: wobei W^p die Gesamtzahl der als p klassifizierten Wörter, W\_s die Gesamtzahl der Wörter in Klasse s und W die Gesamtstichprobengröße ist.  
6. Ein Chi-Quadrat-ähnliches Maß (in der Quelle als MSE bezeichnet) wird berechnet:  
7. Die **Kennzahl ist Cramér's V**, eine normalisierte Version dieses Werts: wobei N die Gesamtstichprobengröße, k die Anzahl der Spalten und r die Anzahl der Zeilen ist.[![][image4]](https://www.codecogs.com/eqnedit.php?latex=%20V%3D0%20#0) bedeutet perfekte Unabhängigkeit (kein Bias), [![][image5]](https://www.codecogs.com/eqnedit.php?latex=V%3D1#0) bedeutet perfekte Assoziation (maximaler Bias).

#### **4.3. Schicht-spezifische Analyse (Logit Lens)**

**Konzept:** Diese diagnostische White-Box-Technik untersucht, *in welcher Transformer-Schicht* (Layer) Bias entsteht oder verstärkt wird. Sie nutzt die "Logit Lens", um die Repräsentationen *jeder* Schicht zu interpretieren.  
**Mathematische Grundlage:** Der Ansatz wendet Grey-Box-Metriken (siehe Abschnitt 5\) auf die Zwischenzustände an :

1. Ein Input (z. B. ein Satzpaar aus CrowS-Pairs) wird durch das Modell propagiert.  
2. Für jede Transformer-Schicht i (von 1 bis N) wird der *hidden state* (Aktivierungsvektor) [![][image6]](https://www.codecogs.com/eqnedit.php?latex=h_i#0) am letzten Token-Position extrahiert.  
3. Dieser Vekto[![][image7]](https://www.codecogs.com/eqnedit.php?latex=r%20h_i#0) wird durch die finale "unembedding"-Matrix (die Schicht, die Vektoren zurück in Logits über das Vokabular projiziert) des Modells geleitet.  
4. Dies erzeugt schicht-spezifische Logits (Wahrscheinlichkeitsverteilungen für das nächste Token) für jede Schicht i.  
5. Eine Grey-Box-Bias-Kennzahl (z. B. der PPL-Score aus 5.1) wird nun *separat für die Logits jeder Schicht* berechnet.  
6. Das Ergebnis ist ein Vektor von Bias-Scores [![][image8]](http://www.texrendr.com/?eqn=Das%20Ergebnis%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)s Ergebnis ist ein Vektor von Bias-Scores [![][image9]](http://www.texrendr.com/?eqn=s%20Ergebnis%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)Ergebnis ist ein Vektor von Bias-Scores [![][image10]](http://www.texrendr.com/?eqn=Ergebnis%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)gebnis ist ein Vektor von Bias-Scores [![][image11]](http://www.texrendr.com/?eqn=gebnis%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)bnis ist ein Vektor von Bias-Scores [![][image12]](http://www.texrendr.com/?eqn=bnis%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)is ist ein Vektor von Bias-Scores [![][image13]](http://www.texrendr.com/?eqn=is%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0) ist ein Vektor von Bias-Scores [![][image14]](http://www.texrendr.com/?eqn=%20ist%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)st ein Vektor von Bias-Scores [![][image15]](http://www.texrendr.com/?eqn=st%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0) ein Vektor von Bias-Scores [![][image16]](http://www.texrendr.com/?eqn=%20ein%20Vektor%20von%20Bias-Scores%20%24%24#0)in Vektor von Bias-Scores [![][image17]](http://www.texrendr.com/?eqn=in%20Vektor%20von%20Bias-Scores%20%24%24#0) Vektor von Bias-Scores [![][image18]](http://www.texrendr.com/?eqn=%20Vektor%20von%20Bias-Scores%20%24%24#0)ektor von Bias-Scores [![][image19]](http://www.texrendr.com/?eqn=ektor%20von%20Bias-Scores%20%24%24#0)tor von Bias-Scores [![][image20]](http://www.texrendr.com/?eqn=tor%20von%20Bias-Scores%20%24%24#0)r von Bias-Scores [![][image21]](http://www.texrendr.com/?eqn=r%20von%20Bias-Scores%20%24%24#0)von Bias-Scores [![][image22]](http://www.texrendr.com/?eqn=von%20Bias-Scores%20%24%24#0)n Bias-Scores [![][image23]](http://www.texrendr.com/?eqn=n%20Bias-Scores%20%24%24#0)Bias-Scores [![][image24]](https://www.codecogs.com/eqnedit.php?latex=%2C%20der%20den%20Verlauf%20des%20Bias%20durch%20die%20Tiefe%20des%20Modells%20zeigt.#0)

### **5\. Grey-Box: Wahrscheinlichkeitsbasierte Metriken**

Diese Metriken erfordern den Zugriff auf die Logit-Ausgaben des Modells und sind fundamental für Benchmarks wie CrowS-Pairs und StereoSet.

#### **5.1. Pseudo-Log-Likelihood (PPL) für kontrafaktische Paare**

**Konzept:** Dies ist die am weitesten verbreitete Methode zur Messung von Stereotypen. Sie testet, ob ein Modell einen stereotypen Satz (S\_{st}) für wahrscheinlicher hält als einen semantisch äquivalenten, aber anti-stereotypen Satz (S\_{at}).  
**Anwendung auf Benchmarks:** CrowS-Pairs, StereoSet.  
**Mathematische Grundlage:**

1. Ein Satzpaar wird bereitgestellt:  
   * S\_{st}: "Der Mann ist ein Ingenieur."  
   * S\_{at}:"Die Frau ist eine Ingenieurin."  
2. Die Pseudo-Log-Likelihood (PPL) wird für jeden Satz berechnet. Dies ist die Summe der bedingten Log-Wahrscheinlichkeiten der Tokens. In wird dies als "All Unmasked Likelihood" (AUL) formalisiert, wobei U die unveränderten Tokens (z. B. "ist ein Ingenieur") und M die geänderten, geschützten Attribut-Tokens (z. B. "Mann" / "Frau") sind: (Die Wahrscheinlichkeit jedes Tokens u\_i im Kontext U, gegeben die vorhergehenden Tokens und das Attribut M).  
3. Die **Kennzahl (CrowS-Pairs Score / Bias Score)** ist der Prozentsatz der Paare, bei denen das Modell den stereotypen Satz S\_{st} als wahrscheinlicher (höhere PPL) einstuft als den anti-stereotypen Satz S\_{at} : wobei N die Anzahl der Satzpaare und \\mathbb{I} die Indikatorfunktion ist (1 wenn wahr, 0 wenn falsch).  
4. **Interpretation:** Ein Score von 50\\% (oder 0.5) gilt als ideal (unvoreingenommen), da das Modell keine Präferenz zeigt. Ein Score von 100\\% (oder 1.0) ist maximal voreingenommen.

#### **5.2. Context Association Test (CAT)**

**Konzept:** CAT misst die Assoziation in die entgegengesetzte Richtung des PPL. Es misst die Wahrscheinlichkeit des *geschützten Attributs* M, gegeben einen *neutralen Kontext* U.  
**Mathematische Grundlage:** Verwendet im StereoSet-Benchmark :  
**Beispiel:** Gegeben der Kontext U \= "Die Person, die als Ingenieur arbeitet, ist...", misst CAT die Wahrscheinlichkeit [![][image25]](https://www.codecogs.com/eqnedit.php?latex=P\(M_%7Bst%7D%20%3D%20%5Ctext%7B%22er%22%7D%20%7C%20U\)#0) im Vergleich zu[![][image26]](https://www.codecogs.com/eqnedit.php?latex=%20P\(M_%7Bat%7D%20%3D%20%5Ctext%7B%22sie%22%7D%20%7C%20U\)#0).

### **6\. Black-Box: Metriken auf Basis generierter Texte**

Dies ist der wichtigste und universellste Methodensatz für die geplante Plattform, da er auf alle Modelle, insbesondere auf kommerzielle APIs, anwendbar ist.

#### **6.1. Klassifikator-basierte Metriken**

**Konzept:** Der generierte Text des LLM wird als Input für ein oder mehrere *sekundäre* Klassifikationsmodelle (Hilfsmodelle) verwendet, um dessen Eigenschaften zu bewerten. Dies ist ein hochgradig automatisierbarer Ansatz.  
**Anwendungen und Kennzahlen:**

1. **Toxizität & Sentiment:** Der LLM wird mit Prompts, die verschiedene soziale Gruppen erwähnen, stimuliert (z. B. "Erzähl mir etwas über \[Gruppe X\]-Menschen."). Die generierte Antwort wird durch einen Toxizitäts-Klassifikator (z. B. Detoxify oder PerspectiveAPI) und einen Sentiment-Klassifikator (z. B. TextBlob ) geleitet.  
   * **Kennzahl:** Die Differenz der durchschnittlichen Scores (z. B. mittlere Toxizität) zwischen den Ausgaben für Gruppe [![][image27]](https://www.codecogs.com/eqnedit.php?latex=G_i#0) und Gruppe [![][image28]](https://www.codecogs.com/eqnedit.php?latex=G_j#0) .  
2. **Demographic Parity Difference (DPD):** Formalisiert in und. Wenn [![][image29]](https://www.codecogs.com/eqnedit.php?latex=c\(%5Ccdot\)#0) der Score des Hilfsklassifikators (z. B. Sentiment) für die generierte Antwort über eine Gruppe G ist:  
3. **"Regard" Score:** Ein spezialisierter Klassifikator, der trainiert wurde, um die "soziale Konnotation" (positiv, negativ, neutral) gegenüber einer Gruppe in einem Text zu messen.

#### **6.2. Verteilungsbasierte Metriken**

**Konzept:** Dieser Ansatz misst Bias als die *Abweichung der Wortverteilung* im generierten Text von einer *Referenzverteilung*, die als "unbiased" oder "real" angenommen wird. Die Studie von Fang et al. bietet hierfür eine vollständige Black-Box-Methodik.  
**Mathematische Grundlage:**

1. **Referenz-Datenbank:** Es wird ein Referenzkorpus o (z. B. Artikel von Reuters, New York Times) als "unbiased" Baseline definiert.  
2. **Generierung:** Das zu testende LLM L generiert einen Textkorpus h, indem es mit denselben Überschriften oder Themen wie im Referenzkorpus o gepromptet wird.  
3. **Verteilung [![][image30]](https://www.codecogs.com/eqnedit.php?latex=%20f_o#0) (Original):** Für den Referenzkorpus o wird die Wahrscheinlichkeitsverteilung f\_o von gruppen-spezifischen Wörtern berechnet (z. B. [![][image31]](https://www.codecogs.com/eqnedit.php?latex=f_o\(%5Ctext%7Bm%C3%A4nnlich%7D\)%20%3D%200.6%2C%20f_o\(%5Ctext%7Bweiblich%7D\)%20%3D%200.4\).#0)  
4. **Verteilung [![][image32]](https://www.codecogs.com/eqnedit.php?latex=f_h%5EL%20#0)(Modell):** Dieselbe Verteilung[![][image33]](https://www.codecogs.com/eqnedit.php?latex=%20f_h%5EL#0) wird für den generierten Textkorpus h des LLM L berechnet.  
5. **Kennzahl: Durchschnittliche Wasserstein-Distanz:** Der Bias des LLM L wird als die durchschnittliche Wasserstein-Distanz (auch Earth Mover's Distance) zwischen den Verteilungen gemessen. Die Wasserstein-Distanz[![][image34]](https://www.codecogs.com/eqnedit.php?latex=%20W\(f_h%5EL%2C%20f_o\)#0) misst, wie viel "Verteilungsmasse" verschoben werden muss, um[![][image35]](https://www.codecogs.com/eqnedit.php?latex=%20f_h%5EL#0) in [![][image36]](https://www.codecogs.com/eqnedit.php?latex=%20f_o#0) zu überführen. Ein höherer [![][image37]](https://www.codecogs.com/eqnedit.php?latex=%5Coverline%7BW%7D%5EL-Score#0) bedeutet eine größere Abweichung von der Referenz und damit einen stärkeren Bias.

#### **6.3. Semantische Bewertung**

Nicht Teil des aktuellen Prototyps.

## **TEIL III: ERFORDERLICHE DATENGRUNDLAGEN (DIE "DATENBANKEN")**

Die in Teil II beschriebenen Metriken sind ohne spezifische "Fragensets" und Wortlisten nicht anwendbar. Eine automatisierte Plattform erfordert die Integration von drei fundamental unterschiedlichen Arten von Datenbanken.

### **7.1. Datenbanktyp 1: Wortlisten (für White-Box-Metriken)**

* **Zweck:** Dienen als Attribut-Mengen (z. B. X, Y) und Ziel-Mengen (z. B. A, B) für geometrische Metriken wie WEAT/SEAT und für das Training/Testen des Klassifikators bei der Cramér's V-Methode.  
* **Inhalt:** Listen von Wörtern, die geschützte Attribute (z. B. [![][image38]](https://www.codecogs.com/eqnedit.php?latex=male_names_de%2C%20female_names_de%2C%20christian_terms_de%2C%20muslim_terms_de\)#0) und stereotype Attribute (z. B. [![][image39]](https://www.codecogs.com/eqnedit.php?latex=career_words_de%2C%20family_words_de%2C%20science_words_de#0)) repräsentieren.  
* **Quelle:** Müssen sorgfältig aus der existierenden Literatur (z. B. ) extrahiert, übersetzt und validiert werden.

### **7.2. Datenbanktyp 2: Kontrafaktische Benchmarks (für Grey-Box-Metriken)**

* **Zweck:** Dienen als Input für PPL- und CAT-Metriken.  
* **Format:** Paare [![][image40]](https://www.codecogs.com/eqnedit.php?latex=\(S_%7Bst%7D%2C%20S_%7Bat%7D\)#0) oder Tupel von Sätzen, die sich typischerweise nur durch die Ersetzung eines geschützten Attributworts unterscheiden.  
* **Katalog der Benchmarks:**  
  * **CrowS-Pairs:** Enthält 1.508 Satzpaare zur Messung von Stereotypen in 9 Bias-Kategorien.  
  * **StereoSet:** Misst Bias in Intra-Satz- und Inter-Satz-Kontexten, indem es eine stereotype, eine anti-stereotype und eine sinnlose Option vergleicht.  
  * **WinoBias / Winogender:** Fokussiert auf Gender-Bias in der Koreferenz-Auflösung (z. B. "Der Arzt... er" vs. "Die Ärztin... sie").  
  * **Bias-STS-B:** Eine Modifikation des Semantic Textual Similarity Benchmark, um zu testen, ob Modelle kontrafaktische Paare fälschlicherweise als semantisch unähnlich bewerten.  
  * **PANDA:** Ein großer Datensatz von Text-Perturbationen über Gender, Ethnie und Alter.

### **7.3. Datenbanktyp 3: Prompt-Sammlungen (für Black-Box-Metriken)**

* **Zweck:** Dienen als Input für alle generativen Tests.  
* **Katalog der Benchmarks:**  
  * **BOLD (Bias in Open-Ended Language Generation Dataset):** Enthält 23.679 Prompts, die verschiedene Demografien erwähnen, um die Fairness von Open-Ended-Generation zu testen.  
  * **RealToxicityPrompts:** 100.000 Prompts (sowohl toxisch als auch nicht-toxisch), um die Generierung von Toxizität als Reaktion auf verschiedene Kontexte zu messen.  
  * **Bias Attack Instructions:** Eine kuratierte Sammlung von Prompts (z. B. aus der Literatur), die entwickelt wurden, um impliziten Bias zu testen.  
  * **BBQ (Bias Benchmark for QA):** Ein Frage-Antwort-Datensatz, der Stereotypen in *zweideutigen* Kontexten misst.  
  * **EquityMedQA:** Ein spezialisierter, adversarischer Datensatz für den medizinischen Bereich, der Fragen mit "biased premise" (voreingenommenen Annahmen) enthält, um die Robustheit von Gesundheits-LLMs zu testen.

Die folgende Tabelle fasst die essenziellen Benchmarks zusammen, die als Grundlage für die Datenbanken der Plattform dienen sollten.  
**Tabelle 1: Essenzielle Benchmarks für die automatisierte Evaluierungsplattform**

| Benchmark | Gemessener Bias | Methodik (Kennzahl) | Erf. Zugriff |
| :---- | :---- | :---- | :---- |
| **CrowS-Pairs** | 9 Typen (Stereotypen) | PPL Score (Grey-Box) | Logits |
| **StereoSet** | 4 Typen (Stereotypen) | PPL Score / CAT (Grey-Box) | Logits |
| **BOLD** | Allg. Demografien | Klassifikator-Metriken (Black-Box) | API (Text-Out) |
| **RealToxicityPrompts** | Toxizität | Toxizitäts-Klassifikator (Black-Box) | API (Text-Out) |
| **Bias Attack Instructions** | 9 Typen (Implizit) | Manuelle/qualitative Auswertung | API (Text-Out) |
| **EquityMedQA** | Health Equity | Manuelle/qualitative Auswertung | API (Text-Out) |

## **TEIL IV: OPERATIONALISIERUNG EINER AUTOMATISIERTEN EVALUIERUNGSPLATTFORM**

Dieser Abschnitt synthetisiert die extrahierten Methoden zu einem kohärenten Entwurf für die "prototypische Implementierung" der Plattform.

### **8\. Architekturkonzept und Datenbankschema**

Die heterogenen Zugriffsebenen (Black-Box vs. White-Box) erfordern zwingend eine modulare Architektur.

* **Modul 1: Black-Box API-Evaluator:**  
  * **Zweck:** Anbindung von Provider-APIs   
  * **Funktionalität:** Implementiert *ausschließlich* die Metriken aus Abschnitt 6 (Klassifikator-basiert, Verteilungs-basiert). Nutzt die Prompt-Datenbanken (Typ 3).  
* **Modul 2: White/Grey-Box Local-Evaluator:**  
  * **Zweck:** Anbindung lokal geladener Open-Source-Modelle (z. B. über Hugging Face Transformers).  
  * **Funktionalität:** Implementiert *alle* Metriken (Abschnitte 4, 5 und 6\) und nutzt *alle* Datenbanktypen (Wortlisten, Kontrafaktische Paare, Prompts).

Eine kritische Erkenntnis aus der Forschung ist die **Fragilität von Bias-Metriken**; kleinste Implementierungsdetails können zu drastisch unterschiedlichen Ergebnissen führen. Daher sollte die Plattform Metriken nicht von Grund auf neu implementieren. Sie sollte stattdessen als **Orchestrierungs-Framework** dienen, das existierende, validierte Open-Source-Bibliotheken (z. B. LangFair , Hugging Face evaluate-bias) kapselt, deren Ausführung automatisiert und die Ergebnisse standardisiert.  
Für die "kontinuierliche Evaluation" ist ein relationales Datenbankschema zur Speicherung der Ergebnisse über die Zeit erforderlich:

* Models: (Model\_ID, Name, Provider, Version\_Hash, Access\_Level)  
* Benchmarks: (Benchmark\_ID, Name, Type, Bias\_Domain)  
* Metrics: (Metric\_ID, Name, Formula\_Reference, Access\_Level)  
* Evaluations: (Eval\_ID, Timestamp, Model\_ID, Benchmark\_ID)  
* Results: (Result\_ID, Eval\_ID, Metric\_ID, Score, Raw\_Output\_Sample)

### **9\. Multi-dimensionale Scoring- und Aggregationsmethoden**

Die Anforderung einer "multi-dimensionalen Bias-Bewertung und Aggregationsmethode" muss mit wissenschaftlicher Sorgfalt behandelt werden. Die Forschung zeigt, dass die Aggregation aller Bias-Formen zu einem *einzigen Score* (z. B. "Gesamt-Bias: 7/10") wissenschaftlich unhaltbar und irreführend ist, da er die schwachen Korrelationen zwischen verschiedenen Bias-Typen und Metriken ignoriert.  
Eine validere Lösung ist die **Profil-basierte Bewertung**, bei der Scores zu aussagekräftigen Vektoren aggregiert werden, die ein "Bias-Profil" eines Modells erstellen:

1. **Profil 1: Aggregation nach Bias-Typ:** Ein Vektor von Scores, der die Leistung des Modells über verschiedene Dimensionen vergleicht: [![][image41]](https://www.codecogs.com/eqnedit.php?latex=Score_Vektor%20%3D#0)  
2. **Profil 2: Aggregation nach Metrik-Typ:** Ein Vektor, der die Robustheit des Modells gegenüber verschiedenen Messmethoden zeigt: [![][image42]](https://www.codecogs.com/eqnedit.php?latex=Score_Vektor%20%3D#0)  
3. **Profil 3: Aggregation nach Anwendungsfall:** Die Plattform sollte "Use Case"-Vorlagen (z. B. "Medizinischer Assistent", "Kundenservice-Chatbot") definieren. Diese Vorlagen wenden unterschiedliche *Gewichtungen* auf die Metriken an. Im medizinischen Fall ist der Score auf EquityMedQA wichtiger als der Score auf WinoBias.

### **10\. Visualisierungskonzepte für eine automatisierte App**

Basierend auf den extrahierten Kennzahlen und der Anforderung einer "automated app", sind folgende Visualisierungen für ein Dashboard der Plattform zentral:

1. **Multi-Achsen-Bias-Profil (Radar-Chart):**  
   * **Zweck:** Direkte Visualisierung des "Profils 1 (By Bias Type)" (siehe 9.1).  
   * **Design:** Ein Radar-Diagramm, bei dem jede Achse einen Bias-Typ (Gender, Race, Religion...) darstellt. Die aufgespannte Fläche repräsentiert die "Bias-Signatur" eines Modells. Mehrere Modelle (z. B. OpenAI vs. Deepseek) können als überlagerte, farbige Flächen für einen direkten Vergleich dargestellt werden.  
2. **Bias-Evolution (Zeitreihen-Liniendiagramm):**  
   * **Zweck:** Erfüllung der Kernanforderung der "kontinuierlichen" Evaluation.  
   * **Design:** X-Achse: Zeit (Datum der Evaluation). Y-Achse: Ein spezifischer, ausgewählter Bias-Score (z. B. "CrowS-Pairs Score"). Verschiedene farbige Linien repräsentieren verschiedene Modelle (z. B. "GPT-4-Turbo" vs. "GPT-4-Omni"), um deren Bias-Entwicklung über Updates hinweg zu verfolgen.  
3. **Schicht-spezifische Bias-Analyse (Liniendiagramm):**  
   * **Zweck:** Detaillierte White-Box-Diagnose (nur für lokale Modelle).  
   * **Design:** Basierend auf der Logit Lens-Methode. X-Achse: Transformer-Schicht (von 1 bis N). Y-Achse: Bias-Score (z. B. PPL-Score). Diese Grafik zeigt visuell, *wo* im Modell (in welchen Schichten) Bias entsteht oder verstärkt wird.  
4. **Verteilungs-Vergleich (Überlagerte Histogramme):**  
   * **Zweck:** Visualisierung der Black-Box-Methode der Verteilungs-Distanz.  
   * **Design:** Zwei überlagerte Histogramme oder Dichte-Plots. Plot 1 (blau): Wortverteilung im Referenzkorpus (z. B. "NYT"). Plot 2 (rot): Wortverteilung im generierten Text des LLM. Die visuelle Diskrepanz zwischen den Kurven *ist* der Bias (quantifiziert durch die Wasserstein-Distanz).  
5. **Qualitative Ergebnis-Tabelle (Interpretierbarkeit):**  
   * **Zweck:** Anzeige hoch-interpretierbarer, qualitativer Ergebnisse (manuelle Stichprobenprüfung).  
   * **Design:** Eine interaktive Tabelle mit den Spalten: Prompt, Modell-Antwort, Notizen/Annotationen.

## **TEIL V: FORSCHUNGSHORIZONTE UND META-ANALYSE (ADVANCED TOPICS)**

Eine robuste Plattform muss nicht nur aktuelle Methoden implementieren, sondern auch deren inhärente Schwächen und die Grenzen der Forschung berücksichtigen.

### **11\. Herausforderung 1: Meta-Bias in Evaluationsverfahren**

Fortgeschrittene semantische Evaluationsverfahren können selbst anfällig für "Meta-Bias" sein.

* **Problem 1: Position Bias:** Der "Judge" (z. B. GPT-4) bewertet zwei Antworten (A und B) nicht objektiv. Es besteht eine systematische Tendenz, die Antwort an Position 1 zu bevorzugen. Die Bewertung Judge(A, B) liefert ein anderes Ergebnis als Judge(B, A).  
* **Problem 2: Superficial Quality Bias:** Der "Judge" bevorzugt "oberflächliche Qualität" (z. B. Ausführlichkeit, Eloquenz, Formalität) gegenüber der eigentlichen *Instruktionstreue* oder *Korrektheit*. Eine lange, eloquent geschriebene, aber voreingenommene Antwort kann fälschlicherweise besser bewertet werden als eine kurze, neutrale, aber korrekte Antwort.

**Lösung für die Plattform:** Die automatisierte Plattform *muss* diese Meta-Biases mitigieren, um valide Ergebnisse zu liefern:

1. **Balanced Position Calibration:** Wie in vorgeschlagen, muss die Plattform bei jedem paarweisen Vergleich (A vs. B) die "Judge"-Bewertung *zweimal* durchführen: einmal mit der Reihenfolge Prompt(A, B) und einmal mit Prompt(B, A). Die Ergebnisse müssen anschließend aggregiert werden (z. B. Durchschnitt oder "Win"-Zählung), um den Position Bias zu eliminieren.  
2. **Kontrastive Benchmark-Tests:** Die Datenbank der Plattform (Teil III) muss Benchmarks wie LLMBar enthalten, die *gezielt* oberflächliche Qualität gegen Instruktionstreue testen.

### **12\. Herausforderung 2: Messung von unvorhergesehenem (Unanticipated) Bias**

Die meisten Metriken (z. B. CrowS-Pairs) testen nur *bekannte* Bias-Achsen (Gender, Race etc.). Die Plattform sollte jedoch idealerweise auch in der Lage sein, "unanticipated" (unvorhergesehenen) Bias gegen Gruppen zu erkennen, die nicht explizit in den Benchmarks definiert sind.  
**Lösungsansatz:** Nutzung von **Uncertainty Quantification (UQ)**.

* **Konzept:** Die Hypothese ist, dass LLMs bei Anfragen zu sozialen Gruppen, für die sie nur spärliche oder voreingenommene Daten haben, eine höhere *Unsicherheit* in ihren Vorhersagen zeigen.  
* **Operationalisierung (Konzept für die Plattform):**  
  1. Erstelle eine Datenbank mit Entitäten aus Nischengruppen.  
  2. Stelle dem LLM Fragen zu diesen Gruppen.  
  3. **Kennzahl (Grey-Box):** Messe die *Unsicherheit* der Antwort, z. B. die Entropie der Logit-Verteilung des nächsten Tokens.  
  4. **Kennzahl (Black-Box):** Messe die *Varianz* der Antworten über mehrere Samples bei einer Temperatur T \> 0\.  
  5. Ein signifikanter Anstieg der Unsicherheit bei bestimmten Gruppen signalisiert eine Repräsentationslücke oder potenziellen "unanticipated bias".

### **13\. Zusammenfassung und wissenschaftliche Einordnung**

Die rigorose Analyse der Forschungsliteratur liefert ein klares Bild für die Konzeption einer automatisierten Bias-Evaluierungsplattform. Die zentrale Erkenntnis ist, dass ein einzelner "Bias Score" nicht existiert und wissenschaftlich irreführend ist.  
Eine valide, kontinuierliche Evaluierung erfordert zwingend einen multi-metrischen und multi-methodischen Ansatz, der die unterschiedlichen Zugriffsebenen (White-, Grey-, Black-Box) berücksichtigt.

1. **Kennzahlen und Mathematik:** Die robustesten Kennzahlen sind der **PPL-Score** für Grey-Box-Szenarien (Messung stereotyper Präferenzen) und die **Wasserstein-Distanz** für Black-Box-Vergleiche (Messung der Abweichung von einer Referenz-Verteilung).  
2. **Methoden-Vergleich:** Black-Box-Methoden sind am universellsten einsetzbar und erfassen impliziten Bias, den RLHF-Modelle sonst verbergen. White-/Grey-Box-Methoden (insb. Logit Lens und Cramér's V) bieten tiefere diagnostische Einblicke, *wo* Bias entsteht und *wie* er strukturiert ist, sind aber auf Open-Source-Modelle beschränkt.  
3. **Operationalisierung:** Die Plattform muss als Orchestrierungs-Framework konzipiert werden, das validierte Benchmarks (z. B. CrowS-Pairs, BOLD) gegen verschiedene Modell-Module (API vs. Lokal) ausführt. Die Ergebnisse müssen als multi-dimensionale "Bias-Profile" (visualisiert über Radar-Charts und Zeitreihen) aggregiert werden.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAANBAMAAAAzjdFVAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEM3v3burMomZRCJ2VGaYzhOJAAAAEklEQVR4XmP8z4AOmNAFhqEQAAKbARn/MowcAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAALBAMAAAD/6NLGAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIs3v3burVGZ2mUQQMon2aNJoAAAAFElEQVR4XmP8z4AJmNAFQGBICwIA7+4BFSyPkMoAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAALBAMAAAD/6NLGAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIs3v3burVGZ2mUQQMon2aNJoAAAAFElEQVR4XmP8z4AJmNAFQGBICwIA7+4BFSyPkMoAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAANBAMAAABvB5JxAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAECJEVHaJq7vd75nNZjIrqulnAAAAEUlEQVR4XmP8zwACTGCSuhQAXqABGU9CYrwAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANBAMAAABbflNtAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAMonN792ZVES7Iqt2EGZy3n0PAAAAEUlEQVR4XmP8zwADTHDWwDEBlUQBGaJlEDEAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAMBAMAAACtsOGuAAAAMFBMVEX////Kysq/v79hYWFUVFRGRkY4ODgoKCgYGBgAAAAAAAAAAAAAAAAAAAAAAAAAAADj6CahAAAAAXRSTlMAQObYZgAAAEFJREFUeF5j/M/AwMD0yeArAwvjBYYCJl4GhglMP0FifxgcGJi4GQwYmBgYmiYACb4Cps8MQCWsDF8nMHF892YAAJY2DUUs9FTYAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAY0AAAAPCAMAAAAvdmOEAAADAFBMVEVHcEwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADR7hC8AAAAFXRSTlMAoi6S2wfN9egQGSOxv1BcRDlphHelSmZZAAAFhklEQVR4Xs1Y6XbiOgx2liYpEAIEvf8Dpju0QxmWq9V2TKZ35k9nvnMa27ItybIki2YuArghHhrAfbyntGLj3OuRJ2831ctbmqJ/zC/SmxQG0kzMBMxnX8yPuN6IUPbO7Q4xWQADjJffbBfi26f1kvl4PfcnGfw+vLZjTJKVuBpTBds6pRggMJvmOuMvfbrJBYRfTrhkjgZNoJRGk8k7P0GYO0j5pmNGrtSIryGmcP92SUB9o6kqaFif49HvIU8JhEzd5xZv/+cuH/ytkcFrMuPRPqeUr3AIYXC/c45vm5EoMntPNSt2CYGhoe0O7YicAB5SSorPVJxLjFk+xaPfwvLPtmRBHny1c/KODc0ppfwCWXJtZN35mPQVNhPpzHXWws1djZS+4l/91RGnEDjkll3aBj8VEqjFs7stt2u4p0nnFkKXHPUmfbfkUMtGwbmVhXFEbn9oB4kciBuVCnJKWQsOl7VL7q9UmqukI4KifIMabrhXF/hpnu5AzwRui9dWaDCAsHPuUVvl0KfZRAkro1eSPjrJoWQYQv/q9K4soYLr99RuA8Ol5BrV3A7igAyakTWJymasklBCHi3xkb+KkigsbGgfdC2ynhD8VGh7uY6WVQbJrVXgMWror0RVFtpnZaEoioYoTh8YiBIsHQJUsLHomBLGNMkXbGPPxsl1GDPpdWpcP2EE0Aci/mPcW1cbvKt1F9asvSuEJXKF4MqCzkhEYsKT/WbE3mKDyokKY6+nwZH8dnDD3vmHjuZzTLozikCPTnMfzS5wb37P6bjhYgs7lxdi5tLMU1kGWb+QZoy5uNr5fD7sSTfhTEzO9ICQg+5l7Cz+5xd9XKh8QZcvyRUHMTvoExjYSBhb0VG6isK0OpolWpGvBFb+oJGg8c9OQlHr7GYRgOH3XLnaHrnywvtXP7EvxiP2xGZwJ0p9rMoPe7TycS6LX/EcF+e7pn1/lSyCWHHwkfSmxarwQ41LuGb25KKwpj3sKAcowcC3RYbqVXogL5EKKOzEM0ktSDZZ22BNUyfMXqoLK0dWmDF3paIVTk/Be+jKSg5/tlPjrlJY2IITKwGo+FkkN6pH015VFbJbhxru+QJUwwWKy654Qb1SBUuc3O5k/xP272jxiq1x8JpLAd1cHatC+yGpVPQ2QqVxkHw40wSPzQK31K/H8KQJ1wcMPfIARHeMChdFVONaFaKRJ6jJ41QYHrRipnY42lEO0Y3gkrw/nOwJRfgH2l4iBvukga9RlA3qqan9KFFcCEFu0BCY0/3ePW2pk4uF6ataZ7RyxPCdPrtd3n+SWqKJzctIfTtX79cUUssr4hOdm5sjogK0xT9BMXxtk1zxM0b1UgR5jtzRIwjmvgdeLbWVMM5cLg6NSXbtLkNh9zpmOgKmvupA1UTASoqpBNmLPcgTiJO/9US1gUIzEsu+ML5hgZyX28tAxq6HRF/NdhTcvSasHFjz54LDZIqtI1cZyNXxRjZ83crVZ7lBXil2Z0RxQpchDVqft63jHbkmk7ORVKwVZwh2m+4Bi8RHXNG+oYkfMBTgkzeAJV3P1KAvULfj8pLyPFDQX1gRJ4WXx/YnPY+UuGITHYVAdRfw1RLWLiNBIeyC2Id7VZf5gzt5hkzs5ZEDekyOC1tJSs3f/FvqX6IS7CflGgopVioY/WIlGvdDJWO/WeNTRLPUUq1JlR9IcQVWF/kFlRSGmeyb63aBpgkgB+iw8pXiTdSiINHFDavuywPgulEKM51f8EZVgbhCjfA60Gdlv79LUjkioNySeqXILVRoXWtlZnxWrCYJkNI2l4U1UZmBar7WGo3G7BfAPx6zWrXlc5Sq1TfhO2X9Gf6aZrO/JBmyycfnX0D037PvBAkdC/4+NYDT978JzVPfDEjN/x8r0G2Oxz5gZQAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHQAAAAQBAMAAAA10UZYAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3vzburmXZEVBBmMiJm6649AAAAHElEQVR4XmP8z0AuYEIXIB6MaiUKjGolCowkrQDqLgEfVcdz0QAAAABJRU5ErkJggg==>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHkAAAAQBAMAAADAL83oAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3vzburmXZEVBBmMiJm6649AAAAHUlEQVR4XmP8z0A++MiELkISGNVNDhjVTQ4YqboBid0CEK9Es4kAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAANBAMAAACEMClyAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIlSJq7vN3e+ZZjIQdkTYCWZ8AAAAFElEQVR4XmP8zwAGH5kgNAMDHRgA4I8CCkwveHoAAAAASUVORK5CYII=>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIlSJq7vN3e+ZZjIQdkTYCWZ8AAAAEUlEQVR4XmP8zwABTFB6gBkAmAEBH3rWMMEAAAAASUVORK5CYII=>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAQBAMAAAAL9CkWAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIlSJu93vzZlEdhCrMma5B41NAAAAFUlEQVR4XmP8zwADH5ngTAaGocIGAHV1AhCS26rlAAAAAElFTkSuQmCC>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAPBAMAAAAizzN6AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARInN71QQdpkiZrur3TKR0mKdAAAAEUlEQVR4XmP8zwACTGCSlhQAbigBHV8/R9QAAAAASUVORK5CYII=>

[image31]: <data:image/png;base64,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>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAARBAMAAAAmgTH3AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARInN71QQdpkiZrur3TKR0mKdAAAAEUlEQVR4XmP8zwABTFB6sDAAoioBIZj5bQIAAAAASUVORK5CYII=>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAARBAMAAAAmgTH3AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARInN71QQdpkiZrur3TKR0mKdAAAAEUlEQVR4XmP8zwABTFB6sDAAoioBIZj5bQIAAAAASUVORK5CYII=>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAARBAMAAAB0ogy8AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAMt3vzburZhCJIplEVHbghmtJAAAAGUlEQVR4XmP8z0AAfGRCF8EEo0qwgxGpBACJ4QISTjBwAwAAAABJRU5ErkJggg==>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAARBAMAAAAmgTH3AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARInN71QQdpkiZrur3TKR0mKdAAAAEUlEQVR4XmP8zwABTFB6sDAAoioBIZj5bQIAAAAASUVORK5CYII=>

[image36]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAPBAMAAAAizzN6AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARInN71QQdpkiZrur3TKR0mKdAAAAEUlEQVR4XmP8zwACTGCSlhQAbigBHV8/R9QAAAAASUVORK5CYII=>

[image37]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFUAAAARBAMAAABeEv0TAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAMt3vzburZhCJIplEVHbghmtJAAAAHUlEQVR4XmP8z0As+MiELoIHjKqFgFG1EDAY1AIAtKkCEo1vKRwAAAAASUVORK5CYII=>

[image38]: <data:image/png;base64,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>

[image39]: <data:image/png;base64,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>

[image40]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAQBAMAAABelcpIAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIrvvMmZ2mc3dEESriVRKaaqQAAAAGUlEQVR4XmP8z4AbfGRCF0EBo7LIYHDKAgClwQIQrhSwjwAAAABJRU5ErkJggg==>

[image41]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAANBAMAAAC3CzZ7AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIlSJq83d73ZEZpm7EDJZAI1hAAAAG0lEQVR4XmP8z0A6YEIXIAaMaoKCUU1QQJYmANMcARngkC2EAAAAAElFTkSuQmCC>

[image42]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAANBAMAAAC3CzZ7AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIlSJq83d73ZEZpm7EDJZAI1hAAAAG0lEQVR4XmP8z0A6YEIXIAaMaoKCUU1QQJYmANMcARngkC2EAAAAAElFTkSuQmCC>