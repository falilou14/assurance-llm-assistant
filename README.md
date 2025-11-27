# 🤖 Assurance LLM Assistant  
Assistant IA spécialisé dans le **NLP (Natural Language Processing)** appliqué au domaine de l’**assurance et des sinistres**.  
Basé sur un **LLM finetuné (LoRA)** et un système **RAG (FAISS)**, il permet de répondre avec précision à des questions métier grâce à une interface **Streamlit** simple et efficace.

---

# 🧠 Rapport avec le NLP

Ce projet est un cas d’étude complet dans le domaine du **Traitement du Langage Naturel (NLP)** :

### 🔹 1. Modèles de Langage (LLM)
Le cœur du projet est un **Large Language Model**, spécialisé grâce à :
- des données métier (documents d’assurance, procédures, lois, conventions, FAQ…)
- un **fine-tuning LoRA** pour adapter le modèle exactement au domaine
- une **quantization 4-bit** pour permettre l'exécution sur machines modestes

### 🔹 2. Fine-Tuning LoRA
LoRA (Low Rank Adaptation) permet d’entraîner *seulement une petite partie du modèle*, rendant le fine-tuning :
- rapide  
- léger  
- accessible sur un laptop  

Le projet démontre :
- la préparation d’un dataset textuel
- l’entraînement supervisé sur des cas métier
- la sauvegarde et le chargement des poids LoRA

### 🔹 3. RAG (Retrieval Augmented Generation)
Le système **RAG** renforce la précision en combinant :
- un LLM finetuné  
- des documents métier indexés avec FAISS  
- des embeddings produits par SentenceTransformers  

Le modèle peut ainsi répondre sur :
- des conventions d’assurance
- des définitions juridiques
- des règles internes de gestion
- des documents que tu fournis

RAG = génération + données métiers → **réponses soutenues par des preuves**.

### 🔹 4. Prompt Engineering
Le projet met en œuvre :
- prompt structuré  
- ajout automatique du contexte RAG  
- format conversationnel adapté  

### 🔹 5. Interface NLP
Avec Streamlit, l’utilisateur interagit :
- en langage naturel  
- avec paramètres ajustables (température, tokens)

Ce projet montre donc la chaîne NLP complète :
**données → fine-tuning → RAG → inférence → interface utilisateur**.

---

# 📌 Use Cases du Projet (Concrets et Métiers)

Ce projet est conçu pour être utilisé immédiatement dans plusieurs situations du domaine assurance :

## 🔹 1. Assistant pour les Agents / Gestionnaires
L’IA peut aider un gestionnaire à :
- comprendre la couverture d’un contrat  
- expliquer une clause obscure  
- estimer si un sinistre est éligible  
- donner les étapes de déclaration d’un incident  
- rappeler la législation applicable  

Exemple :  
> « Le contrat auto formule tiers étendu couvre-t-il un bris de glace arrière ? »

---

## 🔹 2. Aide aux Déclarants de Sinistre (Clients)
L’IA peut servir comme assistant client :
- guider la déclaration d’un sinistre  
- expliquer la franchise  
- déterminer les pièces justificatives nécessaires  
- informer des délais et étapes  

Exemple :  
> « Que dois-je faire si j’ai un dégât des eaux dans mon appartement ? »

---

## 🔹 3. Support pour Centres d’Appel
Elle peut réduire le temps d’attente en répondant à :
- questions administratives  
- conditions de garantie  
- fonctionnement des remboursements  
- suivi de dossiers  

Exemple :  
> « Quelles sont les exclusions courantes en assurance habitation ? »

---

## 🔹 4. Assistant Formation et Documentation
L’IA peut :
- expliquer une loi (Code des assurances)  
- résumer une convention litigieuse  
- simuler des cas pratiques  

Exemple :  
> « Explique-moi la différence entre un dommage corporel et matériel. »

---

## 🔹 5. Analyse des Documents
Avec RAG, elle peut analyser :
- procédures internes  
- contrats  
- FAQ  
- guides de bonne conduite  

Exemple :  
> « Résume-moi les obligations de l’assuré selon ce document. »

---

## 🔹 6. Aide à la Décision pour Experts
Non pas pour remplacer l’expert, mais pour :
- synthétiser des informations  
- proposer des pistes d’analyse  
- comparer des cas  

Exemple :  
> « Liste les éléments à vérifier lors d’un accident de la route non responsable. »

---

# 🏗️ Technologies utilisées

| Technologie | Rôle |
|------------|------|
| **Python** | Langage principal |
| **Transformers (HuggingFace)** | LLM + fine-tuning |
| **PEFT / LoRA** | Fine-tuning léger |
| **BitsAndBytes** | Quantization 4-bit |
| **FAISS** | Index vectoriel RAG |
| **SentenceTransformers** | Embeddings |
| **Streamlit** | Interface utilisateur |
| **PyTorch** | Backend machine learning |
| **GitHub** | Versioning du projet |

---

## 📌 Objectif du projet

Ce projet combine :
- Un **modèle de langage finetuné** (LoRA)
- Un système **RAG** pour injecter du contexte réel (documents assurance)
- Une interface utilisateur **Streamlit**
- Une architecture propre permettant de :
  - réentraîner le modèle,
  - modifier les données RAG,
  - utiliser des modèles légers adaptés à des machines limitées.

---

## 🧠 Fonctionnalités

### ✔ Fine-Tuning LoRA (CPU/GPU)
- Basé sur `transformers` + `peft`
- Support de la quantization 4-bit (optimisation mémoire)
- Sauvegarde des poids LoRA dans `models/`

### ✔ RAG (Retrieval Augmented Generation)
- Embeddings via SentenceTransformers
- Indexation FAISS
- Recherche des documents pertinents
- Injection automatique dans le prompt du modèle

### ✔ Interface Streamlit
- Poser des questions
- Activer/désactiver le RAG
- Ajuster température et nombre de tokens
- Afficher les documents indexés

### ✔ Pipeline d’inférence robuste
- Fonctionne sur CPU uniquement
- Compatible GPU si disponible
- Chargement LoRA sécurisé (sans crash `accelerate`)
- Modèle léger supporté (3B à 7B)


---

## 🚀 Installation

###  Cloner le projet
```bash
git clone https://github.com/<ton-username>/assurance-llm-assistant.git
cd assurance-llm-assistant

### Installer les dépendances
```bash
pip install -r requirements.txt

### Lancer l’interface
```bash
streamlit run app/streamlit_app.py

