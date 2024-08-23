# Description

## Dossier models

Le dossier models contient des classes qui héritent de keras.Model, elles possèdent donc les mêmes propriétés et cela permet de charger tous les modèles en important le fichier. Il y a un sous dosser *variational_models* qui contient globalement les mêmes modèles mais avec leurs versions variationnelles. Dans chacun des fichies se trouvent 3 implémentation correspondant à la version variationnelle 1, 2 & 3 qui sont les différentes positions de la couche variationnelle. Tous les modèles sont disponibles dans leur version 1 (fusion de modalité précoce) et version 2 (fusion tardive).

## Dossier utils

Ce dossier ne contient pour l'instant qu'une implémentation custom de la Quantile Huber Loss (qui hérite de keras.losses.Loss). A terme il pourrait contenir tous les outils au fonctionnement des modèles de Deep Learning.

## Dossier VAE

Le dossier VAE permet d'une part d'entrainer un Variational Autoencoder à reproduire les observations côniques (il faut récupérer au préalable une copie de la mémoire d'un agent) grâce au fichier vae_training. Le fichier VAE.py contient le code du Variational Autoencoder utilisé.
Le fichier interface.py est une implémentation pygame dans laquelle il est possible de régler les valeurs de l'espace latent d'un modèle variationnel. Cela a été conçu davantage pour les modèles à couche variationnelle plutôt que les VAE mais devrait pouvoir s'adapter sans mal.
**A noter** : des modifications d'arborescence sont sans doute à prévoir.

## Dossier processes 

Dossier contenant des classes héritant de threading.Thread et multiprocessing.Process respectivement. La première permet de lancer un Thread d'entrainement, on distinguera 4 types de thread :
- ThreadProcess classiques pour un entrainement de Q-Learning basiques
- DistributionalThreadProcess pour entrainer des modèles par Quantile Régression (la majorité des modèles a été entrainée de cette façon)
- VariationalThreadProcess qui subi quelques modifications du ThreadProcess original pour gérer l'utilisation d'un VAE sur le cône d'observations
- InferenceThreadProcess afin de faire tourner tout type de modèle en inférence (potentiellement sur la carte originale du projet)

En termes de WorkerProcess il n'y a qu'une classe fille qui génère des environnements, récolte les états et applique les actions. Elle permet la parallélisation grâce à multiprocessing car chaque ThreadProcess possède une liste de WorkerProcess.

## Dossier EnergyBoatScenario
Contient notamment la classe d'Environnement, le fichier contenant les fonctions d'affichage et certaines fonctions utilitaires.
Pour répondre aux besoins du Reinforcement Learning, l'environnement comporte une fonction d'observation *get_env_state*, une fonction *step*, une fonction *reset* et une fonction *render*. 

## Fichiers supplémentaires
Une classe ReplayBuffer a été implémentée, elle permet le stockage des transitions et leur sélection aléatoire pour l'Exprience Replay.
Une classe RecordingBuffer assez similaire sert cette fois à enregistrer les courbes d'entrainement. Elle stocke les métriques après chaque épisodes et produit des graphiques selon la fréquence demandée.
Le fichier train.py permet de lancer des entrainements en faisant appels aux classes de thread_process et aux modèles de models. Le fichier test.py quand à lui permet de lancer un InferenceThreadProcess et de stocker les résultats dans un dataframe.