# 🌐 Portfolio – Guide de déploiement Azure Static Web Apps

## Fichiers du projet

```
/
├── index.html                          ← Le site web complet
├── staticwebapp.config.json            ← Config Azure Static Web Apps
├── .github/
│   └── workflows/
│       └── azure-static-web-apps.yml   ← CI/CD automatique
└── README.md
```

---

## 🚀 Déploiement sur Azure for Students – Étape par étape

### Prérequis
- Un email étudiant valide (`.edu` ou email de l'établissement)
- Un compte GitHub gratuit
- Un navigateur web

---

### Étape 1 – Activer Azure for Students

1. Aller sur [azure.microsoft.com/fr-fr/free/students](https://azure.microsoft.com/fr-fr/free/students/)
2. Cliquer sur **"Démarrer gratuitement"**
3. Se connecter avec son **email étudiant**
4. Renseigner les informations sans CB requise ✅
5. Obtenir **100 $ de crédits** valables 12 mois

---

### Étape 2 – Préparer le dépôt GitHub

```bash
# Cloner ou créer un nouveau dépôt
git init mon-portfolio
cd mon-portfolio

# Copier les fichiers du projet
cp index.html .
cp staticwebapp.config.json .

# Premier commit
git add .
git commit -m "Initial portfolio"

# Pousser sur GitHub
git remote add origin https://github.com/TON_USERNAME/mon-portfolio.git
git push -u origin main
```

---

### Étape 3 – Créer Azure Static Web Apps

1. Se connecter sur [portal.azure.com](https://portal.azure.com)
2. Rechercher **"Static Web Apps"** dans la barre de recherche
3. Cliquer sur **"Créer"**
4. Remplir le formulaire :
   - **Abonnement** : Azure for Students
   - **Groupe de ressources** : `rg-portfolio` (créer nouveau)
   - **Nom** : `portfolio-alex-moreau`
   - **Plan** : **Gratuit**
   - **Région** : West Europe
5. Section **Déploiement** :
   - **Source** : GitHub
   - Autoriser Azure à accéder à votre GitHub
   - **Organisation** : votre compte
   - **Dépôt** : `mon-portfolio`
   - **Branche** : `main`
6. Section **Build** :
   - **Préréglages** : `Custom`
   - **Emplacement de l'application** : `/`
   - **Emplacement de l'API** : (vide)
   - **Emplacement de sortie** : (vide)
7. Cliquer sur **"Vérifier + créer"** puis **"Créer"**

---

### Étape 4 – Déploiement automatique

Azure va :
1. Créer le fichier `.github/workflows/azure-static-web-apps.yml` dans votre repo
2. Lancer un **GitHub Actions** automatiquement
3. Déployer le site en **~2 minutes**

Vous pouvez suivre le déploiement dans :
- **Azure Portal** → votre ressource Static Web Apps → Onglet "Environnements"
- **GitHub** → votre repo → Onglet "Actions"

---

### Étape 5 – Accéder au site

Après le déploiement, votre URL sera de la forme :
```
https://lively-sea-XXXXXX.azurestaticapps.net
```

Vous pouvez aussi ajouter un **domaine custom** gratuit dans :
Azure Portal → Static Web App → Custom domains

---

## 💡 Personnalisation

Pour adapter le site à votre identité, modifier dans `index.html` :

| Élément | Balise / Classe | Valeur à changer |
|---------|----------------|-----------------|
| Votre nom | `<span class="first">` et `<span class="last">` | Alex / Moreau |
| Votre email | `href="mailto:..."` | votre@email.fr |
| Votre formation | `.hero-title` | BTS SIO SLAM |
| Vos compétences | `.skill-item` (%) | Valeurs 0-100 |
| Vos projets | `.project-card` | Titre, description |
| Votre ville | `.contact-item` | Paris, Lyon… |

---

## 📊 Coûts

| Service | Plan | Coût |
|---------|------|------|
| Azure Static Web Apps | Gratuit | 0 €/mois |
| GitHub | Gratuit | 0 €/mois |
| Certificat HTTPS | Inclus | 0 €/mois |
| CDN mondial | Inclus | 0 €/mois |
| **TOTAL** | | **0 €/mois** ✅ |

> Azure for Students offre 100 $ de crédits – les Static Web Apps gratuites n'en consomment pas.

---

## 🔗 Liens utiles

- [Azure for Students](https://azure.microsoft.com/fr-fr/free/students/)
- [Azure Static Web Apps – Doc officielle](https://docs.microsoft.com/fr-fr/azure/static-web-apps/)
- [GitHub Actions](https://github.com/features/actions)
