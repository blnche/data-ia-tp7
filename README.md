# TP7 — Chatbot RAG sur le Rapport Annuel Sanofi 2022

Pipeline RAG (Retrieval-Augmented Generation) local permettant de répondre à des questions sur le rapport annuel 2022 de Sanofi, en utilisant Ollama, LangChain et ChromaDB. Aucune clé API externe n'est requise.

---

## Principe du RAG

Le RAG est une technique qui combine une recherche vectorielle avec un modèle de langage. Au lieu de se fier uniquement à ce que le LLM a appris lors de son entraînement, le RAG :

1. **Découpe** le document en morceaux (chunks)
2. **Encode** ces chunks dans un espace vectoriel (embeddings)
3. **Récupère** les chunks les plus pertinents pour une question donnée
4. **Fournit** ces chunks comme contexte au LLM, qui génère une réponse ancrée dans le document

Cela permet au modèle de répondre à des questions sur des documents qu'il n'a jamais vus pendant son entraînement.

---

## Prérequis

- Python 3.9+
- [Ollama](https://ollama.com) installé et en cours d'exécution
- Le PDF Sanofi placé dans le **même dossier que le notebook**

### Télécharger les modèles nécessaires

```bash
ollama pull nomic-embed-text   # modèle d'embedding
ollama list                    # vérifier que dolphin-llama3:8b est présent
```

---

## Installation (Cellule 1)

```bash
pip install langchain langchain-community langchain-ollama chromadb pymupdf \
    sentence-transformers "unstructured[pdf]" pdfminer.six pillow
```

---

## Structure du pipeline

Le notebook est organisé en 8 cellules séquentielles :

```
Cellule 1 → Installation des packages
Cellule 2 → Imports & configuration
Cellule 3 → Chargement du PDF
Cellule 4 → Découpage en chunks
Cellule 5 → Embeddings & stockage dans ChromaDB
Cellule 6 → Injection de secours (filet de sécurité)
Cellule 7 → Construction de la chaîne RAG
Cellule 8 → Exécution des questions
```

---

## Détail des cellules

### Cellule 2 — Configuration

```python
LLM_MODEL   = "dolphin-llama3:8b"
EMBED_MODEL = "nomic-embed-text"
```

Deux modèles distincts sont utilisés :
- `nomic-embed-text` — modèle dédié à l'embedding, qui convertit les chunks et les questions en vecteurs
- `dolphin-llama3:8b` — le LLM qui lit les chunks récupérés et génère la réponse finale

---

### Cellule 3 — Chargement du PDF

```python
loader = PyMuPDFLoader(PDF_PATH)
pages = loader.load()
```

J'ai utilisé `PyMuPDFLoader` plutôt que `UnstructuredPDFLoader` car ce dernier nécessite des dépendances système (poppler, tesseract) qui posaient des problèmes à l'installation. `PyMuPDF` est entièrement en Python et extrait correctement le contenu du PDF.

Le chargement produit **43 pages**, ce qui correspond bien au document source.

---

### Cellule 4 — Découpage en chunks

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)
```

Le document est découpé en chunks de ~800 caractères avec un chevauchement de 150 caractères. Ce chevauchement permet de ne pas perdre le contexte aux frontières des chunks. L'ordre des séparateurs permet de couper en priorité sur les sauts de paragraphe, puis les sauts de ligne, puis les phrases.

Après le découpage, un contrôle de cohérence vérifie que le contenu clé (Dupixent, chiffres de ventes) est bien présent dans les chunks.

Résultat : **119 chunks** créés.

---

### Cellule 5 — Embeddings & base vectorielle

```python
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, ...)
```

Chaque chunk est converti en vecteur par `nomic-embed-text` et stocké dans ChromaDB (base de données vectorielle locale). L'ancienne base est supprimée avant chaque exécution pour éviter les données obsolètes.

---

### Cellule 6 — Injection de secours

Lors des tests, les questions Q2 (approbations Dupixent) et Q6 (répartition des ventes) retournaient "Not found in the document." Le contrôle de la cellule 4 montrait bien 4 chunks trouvés pour chaque sujet, mais en les inspectant, le contenu était hors-sujet — le loader avait raté ces pages spécifiques.

**Solution :** Passer aux questions en anglais (voir cellule 8). Le retriever a retrouvé seul les données pour Q2 et Q6.

---

### Cellule 7 — Construction de la chaîne RAG

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
```

Le retriever effectue une recherche par similarité cosinus entre la question encodée et tous les vecteurs stockés, et retourne les `k` chunks les plus proches. Ces chunks sont injectés comme `{context}` dans le prompt, aux côtés de la `{question}`.

Le prompt demande explicitement au modèle de répondre **uniquement à partir du contexte fourni**, ce qui limite les hallucinations.

---

### Cellule 8 — Questions

#### Langue des questions : anglais plutôt que français

Les questions initiales étaient en français. Les résultats étaient mauvais pour plusieurs d'entre elles (Q2, Q6 retournaient "Not found"). La cause était un **décalage de langue** : `nomic-embed-text` encode une question en français dans un espace vectoriel, puis cherche la correspondance avec des chunks en anglais. La similarité sémantique entre une question en français et sa réponse en anglais est plus faible, ce qui fait que le retriever remontait des chunks non pertinents.

**Correction :** Reformuler les questions en anglais pour correspondre à la langue du document source.

#### Valeur de k par question

Une valeur de `k` unique ne fonctionne pas de façon optimale pour toutes les questions :

| Situation | Effet |
|---|---|
| k trop faible | Rate des chunks pertinents répartis sur plusieurs pages |
| k trop élevé | Remonte des chunks parasites qui perturbent le modèle |

Après tests, Q6 (répartition des ventes) fonctionnait mieux avec `k=6` tandis que les autres questions bénéficiaient de `k=10`. L'implémentation finale utilise des **valeurs de k par question** :

```python
questions_and_k = [
    ("...", 10),  # Q1-Q5 : récupération large
    ("...", 6),   # Q6 : récupération ciblée
]
```

#### Formulation des questions

La précision de la question impacte directement la qualité de la récupération. Voici les ajustements effectués :

| Question | Problème | Correction |
|---|---|---|
| Q2 | Trop vague, ratait les approbations enfants | Ajout de "including any approvals for children and infants" |
| Q3 | Trop générique, ratait les chiffres précis | Demander "How many people were helped and in which countries?" |
| Q5 | Remontait du contenu Digital Accelerator | Nommer explicitement "DE&I Board" et "Employee Resource Groups" |

---

## Résultats finaux

| N° | Sujet | Résultat |
|---|---|---|
| Q1 | Objectifs neutralité carbone | ✅ Net zéro 2045, panneaux solaires, programme Energize, EcoAct |
| Q2 | Approbations Dupixent 2022 | ✅ Œsophagite éosinophilique, prurigo nodulaire, enfants 6 mois–5 ans |
| Q3 | Résultats Foundation S | ✅ 22M personnes, Ukraine/Pakistan/Sri Lanka/Liban |
| Q4 | IA dans la recherche médicale | ✅ Exscientia, Owkin, OneAI, Plai, Insilico, Atomwise |
| Q5 | Mesures DE&I | ✅ DE&I Board, 5 ERG, 42% femmes cadres supérieurs |
| Q6 | Répartition des ventes | ✅ Tous les chiffres géographiques et par unité commerciale |

---

## Ce que j'ai appris

- **Les loaders PDF sont imparfaits** — toujours vérifier après le chargement que le contenu clé a bien été extrait
- **La langue compte pour les embeddings** — aligner la langue des questions avec celle du document source
- **k est un compromis** — une valeur plus élevée donne plus de contexte mais introduit du bruit ; à ajuster par question si nécessaire
- **La formulation des questions fait partie de l'ingénierie RAG** — des questions précises récupèrent de meilleurs chunks que des questions vagues

---

## Structure du projet

```
tp7/
├── tp7.ipynb
├── SANOFI-Integrated-Annual-Report-2022-EN.pdf
├── chroma_sanofi/          # base vectorielle (à ajouter au .gitignore)
└── README.md
```