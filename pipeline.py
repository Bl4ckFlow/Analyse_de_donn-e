import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectFromModel
import joblib
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TRANSFORMERS PERSONNALISÉS
# =============================================================================

class TelecomPreprocessor(BaseEstimator, TransformerMixin):
    """Transformer personnalisé pour les données de télécommunications"""
    
    def __init__(self, imputation_strategy='median', handle_outliers=True):
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.region_encoder = None
        self.outlier_bounds = {}
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.feature_names_in_ = list(X_df.columns)
        
        # Séparer colonnes numériques et catégorielles
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Préparer les imputeurs
        if numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy=self.imputation_strategy)
            self.numeric_imputer.fit(X_df[numeric_cols])
            
            if self.handle_outliers:
                X_imputed = pd.DataFrame(
                    self.numeric_imputer.transform(X_df[numeric_cols]),
                    columns=numeric_cols
                )
                for col in numeric_cols:
                    Q1 = X_imputed[col].quantile(0.25)
                    Q3 = X_imputed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    self.outlier_bounds[col] = {
                        'lower': Q1 - 1.5 * IQR,
                        'upper': Q3 + 1.5 * IQR
                    }
        
        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(X_df[categorical_cols])
            
            if len(categorical_cols) > 0:
                self.region_encoder = LabelEncoder()
                region_col = categorical_cols[0]
                temp_data = X_df[region_col].fillna('missing')
                self.region_encoder.fit(temp_data)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Imputation numérique
        if numeric_cols and self.numeric_imputer:
            X_df[numeric_cols] = self.numeric_imputer.transform(X_df[numeric_cols])
            
            # Traitement des outliers
            if self.handle_outliers:
                for col in numeric_cols:
                    if col in self.outlier_bounds:
                        bounds = self.outlier_bounds[col]
                        X_df[col] = X_df[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        
        # Imputation et encodage catégoriel
        if categorical_cols and self.categorical_imputer:
            X_df[categorical_cols] = self.categorical_imputer.transform(X_df[categorical_cols])
            
            if self.region_encoder and len(categorical_cols) > 0:
                region_col = categorical_cols[0]
                # Gestion des nouvelles catégories
                def safe_transform(x):
                    try:
                        return self.region_encoder.transform([x])[0]
                    except ValueError:
                        return 0 
                
                X_df[f'{region_col}_encoded'] = X_df[region_col].apply(safe_transform)
                X_df = X_df.drop(columns=[region_col])
        
        self.feature_names_out_ = list(X_df.columns)
        return X_df.values
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_ if self.feature_names_out_ else self.feature_names_in_

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Transformer pour encoder la variable cible en catégories"""
    
    def __init__(self, n_categories=4, method='quantile'):
        self.n_categories = n_categories
        self.method = method
        self.bins_ = None
        self.label_encoder_ = None
        
    def fit(self, y):
        if self.method == 'quantile':
            self.bins_ = np.quantile(y, np.linspace(0, 1, self.n_categories + 1))
        elif self.method == 'equal':
            self.bins_ = np.linspace(y.min(), y.max(), self.n_categories + 1)
        
        # Créer les catégories
        categories = pd.cut(y, bins=self.bins_, labels=False, include_lowest=True)
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(categories)
        return self
    
    def transform(self, y):
        categories = pd.cut(y, bins=self.bins_, labels=False, include_lowest=True)
        return self.label_encoder_.transform(categories)
    

# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================


class TelecomMLPipeline:
    """
    Pipeline complet de Machine Learning pour les données de télécommunications
    """
    
    def __init__(self, df_abs, df_rel):
        self.df_abs = df_abs.copy()
        self.df_rel = df_rel.copy()
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.task_type = None
        self.target_encoder = None
        
        # Pipelines sklearn
        self.preprocessing_pipeline = None
        self.pca_pipeline = None
        self.model_pipelines = {}
        self.best_pipelines = {}
        
        # Résultats
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, target_variable, use_relative=True, create_target_categories=False, 
                 missing_threshold=0.5, test_size=0.2, random_state=42):
        """
        2. Préparation des données : Nettoyage GLOBAL puis séparation
        
        1. Nettoyage global de TOUT le dataset
        2. Analyse des variables et patterns de données manquantes  
        3. Séparation features/target sur données nettoyées
        4. Split train/test avec stratification appropriée
        """
        print("="*60)
        print(" 2. PRÉPARATION DES DONNÉES (VERSION CORRIGÉE)")
        print("="*60)
        
        # Choisir le dataset
        df = self.df_rel.copy() if use_relative else self.df_abs.copy()
        print(f" Dataset utilisé: {'Relatif (per 100 hab.)' if use_relative else 'Absolu (millions)'}")
        print(f" Variable cible: {target_variable}")
        print(f" Données initiales: {df.shape}")
        

        print(f"\n 1. ANALYSE GLOBALE DES DONNÉES MANQUANTES")
        print("-" * 50)
        
        # Calculer le pourcentage de données manquantes par variable
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        missing_analysis = pd.DataFrame({
            'Variable': df.columns,
            'Missing_Count': missing_stats,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Variables avec données manquantes :")
        vars_with_missing = missing_analysis[missing_analysis['Missing_Count'] > 0]
        if not vars_with_missing.empty:
            for _, row in vars_with_missing.head(10).iterrows():
                print(f"   {row['Variable']:25s} | {row['Missing_Count']:4.0f} ({row['Missing_Percentage']:5.1f}%)")
        else:
            print("  Aucune donnée manquante détectée!")
        
        # Identifier les variables à exclure (trop de données manquantes)
        vars_to_exclude = missing_analysis[missing_analysis['Missing_Percentage'] > missing_threshold * 100]['Variable'].tolist()
        if vars_to_exclude:
            print(f"\n Variables à exclure (>{missing_threshold*100}% manquant):")
            for var in vars_to_exclude:
                pct = missing_analysis[missing_analysis['Variable'] == var]['Missing_Percentage'].iloc[0]
                print(f"   - {var}: {pct:.1f}% manquant")
            
            # Exclure ces variables du dataset
            df = df.drop(columns=vars_to_exclude)
            print(f"   Dataset après exclusion: {df.shape}")
        
        # Vérifier que la variable cible est encore présente
        if target_variable not in df.columns:
            raise ValueError(f" Variable cible '{target_variable}' exclue car trop de données manquantes!")
        
        print(f"\n 2. NETTOYAGE INTELLIGENT DES OBSERVATIONS")
        print("-" * 50)
        
        initial_obs = len(df)
        

        df_clean = df.dropna(subset=[target_variable])
        print(f"   Suppression lignes target manquante: {initial_obs} → {len(df_clean)} obs")
        
        missing_per_row = df_clean.isnull().sum(axis=1)
        max_missing_per_row = int(len(df_clean.columns) * 0.8)  # 80% de variables manquantes max
        
        rows_too_missing = missing_per_row > max_missing_per_row
        if rows_too_missing.sum() > 0:
            df_clean = df_clean[~rows_too_missing]
            print(f"   Suppression lignes avec >80% manquant: {len(df_clean)} obs restantes")
        
        print(f" Données nettoyées: {df_clean.shape}")
        print(f" Réduction: {initial_obs} → {len(df_clean)} observations ({len(df_clean)/initial_obs*100:.1f}%)")
        

        print(f"\n 3. ANALYSE DES PATTERNS RESTANTS")
        print("-" * 40)
        
        remaining_missing = df_clean.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0].sort_values(ascending=False)
        
        if len(remaining_missing) > 0:
            print("Données manquantes restantes (à imputer):")
            for var, count in remaining_missing.head(5).items():
                pct = (count / len(df_clean)) * 100
                print(f"   {var:25s} | {count:4.0f} ({pct:5.1f}%)")
            
            # Analyser les correlations entre patterns de données manquantes
            missing_matrix = df_clean.isnull()
            if len(remaining_missing) > 1:
                missing_corr = missing_matrix[remaining_missing.index].corr()
                high_corr_pairs = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_val = missing_corr.iloc[i, j]
                        if abs(corr_val) > 0.5:  # Corrélation forte entre patterns manquants
                            high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))
                
                if high_corr_pairs:
                    print(f"\n Patterns de données manquantes corrélés:")
                    for var1, var2, corr in high_corr_pairs[:3]:
                        print(f"      {var1} ↔ {var2}: {corr:.2f}")
        else:
            print(" Plus de données manquantes à traiter!")
        

        print(f"\n 4. SÉPARATION FEATURES / TARGET")
        print("-" * 35)
        
        # Maintenant on sépare sur des données propres
        self.feature_names = [col for col in df_clean.columns if col != target_variable]
        X = df_clean[self.feature_names].copy()
        y = df_clean[target_variable].copy()
        
        print(f"   Features disponibles: {len(self.feature_names)}")
        print(f"   Observations pour ML: {len(X)}")
        

        if create_target_categories:
            print(f"\n 5. CRÉATION CATÉGORIES POUR CLASSIFICATION")
            print("-" * 45)
            
            # Analyser la distribution de la target continue
            print(f"   Distribution target continue:")
            print(f"      Min: {y.min():.2f} | Max: {y.max():.2f}")
            print(f"      Moyenne: {y.mean():.2f} | Médiane: {y.median():.2f}")
            
            self.target_encoder = TargetEncoder(n_categories=4, method='quantile')
            y_categorical = self.target_encoder.fit_transform(y)
            
            # Analyser la distribution des catégories créées
            unique_cats, counts = np.unique(y_categorical, return_counts=True)
            print(f"   Catégories créées: {len(unique_cats)}")
            for cat, count in zip(unique_cats, counts):
                print(f"      Catégorie {cat}: {count} obs ({count/len(y_categorical)*100:.1f}%)")
            
            y = y_categorical
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
            print(f"\n Tâche: Régression (target continue)")
        
        print(f"\n 6. SÉPARATION TRAIN/TEST")
        print("-" * 30)
        
        # Split intelligent avec stratification si classification
        if self.task_type == 'classification':
            # Vérifier qu'on a assez d'exemples par classe
            unique_vals, counts = np.unique(y, return_counts=True)
            min_count = counts.min()
            
            if min_count < 2:
                print(f" Attention: classe minoritaire avec {min_count} exemple(s)")
                print(f" Pas de stratification possible")
                stratify_param = None
            else:
                stratify_param = y
                print(f" Stratification par classe activée")
        else:
            stratify_param = None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"   Train set: {self.X_train.shape[0]} observations ({(1-test_size)*100:.0f}%)")
        print(f"   Test set:  {self.X_test.shape[0]} observations ({test_size*100:.0f}%)")
        
        # Vérifier l'équilibre si classification
        if self.task_type == 'classification':
            print(f"\n   Distribution train/test par classe:")
            train_dist = pd.Series(self.y_train).value_counts().sort_index()
            test_dist = pd.Series(self.y_test).value_counts().sort_index()
            
            for class_val in unique_vals:
                train_count = train_dist.get(class_val, 0)
                test_count = test_dist.get(class_val, 0) 
                print(f" Classe {class_val}: Train={train_count}, Test={test_count}")
        

        print(f"\n 7. RÉSUMÉ DE LA PRÉPARATION")
        print("=" * 40)
        print(f"  Données finales:")
        print(f"   • Observations utilisables: {len(X)}")
        print(f"   • Features retenues: {len(self.feature_names)}")
        print(f"   • Variables exclues: {len(vars_to_exclude)}")
        print(f"   • Type de tâche: {self.task_type}")
        print(f"   • Split: {len(self.X_train)} train / {len(self.X_test)} test")
        
        print(f"\n Features sélectionnées:")
        for i, feature in enumerate(self.feature_names):
            missing_in_feature = X[feature].isnull().sum()
            missing_pct = (missing_in_feature / len(X)) * 100
            print(f"   {i+1:2d}. {feature:25s} | {missing_pct:5.1f}% manquant")
        
        # Retourner les informations importantes
        return {
            'X_train_shape': self.X_train.shape,
            'X_test_shape': self.X_test.shape,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'n_features_original': len(df.columns) - 1, 
            'n_features_final': len(self.feature_names),
            'n_observations_original': initial_obs,
            'n_observations_final': len(X),
            'variables_excluded': vars_to_exclude,
            'data_reduction_pct': len(X) / initial_obs * 100
        }
        
    def perform_pca(self, n_components=None, variance_threshold=0.95):
        """
        3. Analyse en Composantes Principales (PCA)
        """
        print("="*60)
        print(" 3. ANALYSE EN COMPOSANTES PRINCIPALES (PCA)")
        print("="*60)
        
        # Créer le pipeline de preprocessing + PCA
        preprocessing_steps = [
            ('preprocessor', TelecomPreprocessor()),
            ('scaler', StandardScaler())
         ]
        
        # Déterminer le nombre optimal de composantes si non spécifié
        if n_components is None:
            # Pipeline temporaire pour estimer
            temp_pipeline = Pipeline(preprocessing_steps + [('pca', PCA())])
            temp_pipeline.fit(self.X_train, self.y_train)
            
            explained_variance_ratio = temp_pipeline.named_steps['pca'].explained_variance_ratio_
            cumsum_variance = np.cumsum(explained_variance_ratio)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            print(f" Nombre optimal de composantes pour {variance_threshold*100}% variance: {n_components}")
        
        # Créer le pipeline final avec PCA
        self.pca_pipeline = Pipeline(preprocessing_steps + [('pca', PCA(n_components=n_components))])
        
        # Fit sur les données d'entraînement
        self.pca_pipeline.fit(self.X_train, self.y_train)
        
        # Transformer les données
        X_train_pca = self.pca_pipeline.transform(self.X_train)
        X_test_pca = self.pca_pipeline.transform(self.X_test)
        
        # Analyser les résultats
        pca_component = self.pca_pipeline.named_steps['pca']
        explained_variance = pca_component.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f" Variance expliquée par composante:")
        for i, (var_exp, cum_var) in enumerate(zip(explained_variance[:5], cumulative_variance[:5])):
            print(f"   PC{i+1}: {var_exp:.3f} (cumulé: {cum_var:.3f})")
        
        print(f"\n PCA terminé!")
        print(f"   Réduction: {self.X_train.shape[1]} → {n_components} features")
        print(f"   Variance totale conservée: {cumulative_variance[-1]:.3f}")
        
        # Visualisations
        self._plot_pca_analysis(explained_variance, cumulative_variance, X_train_pca)
        
        return {
            'n_components': n_components,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca
        }
    
    
    def train_models(self, use_pca=False):
        """
        4. Modélisation supervisée : Entraînement des modèles avec pipelines
        """
        print("="*60)
        print(" 4. MODÉLISATION SUPERVISÉE")
        print("="*60)
        
        print(f" Type de tâche: {self.task_type}")
        print(f" Utilisation PCA: {'Oui' if use_pca else 'Non'}")
        
        # Étapes de preprocessing de base
        preprocessing_steps = [
            ('preprocessor', TelecomPreprocessor()),
            ('scaler', StandardScaler())
        ]
        
        # Ajouter PCA si demandé
        if use_pca and self.pca_pipeline is not None:
            pca_component = self.pca_pipeline.named_steps['pca']
            preprocessing_steps.append(('pca', PCA(n_components=pca_component.n_components_)))
        
        # Définir les modèles selon le type de tâche
        if self.task_type == 'regression':
            models_config = {
                'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'Decision_Tree': DecisionTreeRegressor(random_state=42),
                'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
            }
        else:
            models_config = {
                'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
                'Decision_Tree': DecisionTreeClassifier(random_state=42),
                'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
            }
        
        print(f"\n Entraînement des modèles...")
        
        # Créer et entraîner les pipelines
        for name, model in models_config.items():
            print(f"   Entraînement {name}...")
            
            try:
                # Créer le pipeline complet
                pipeline = Pipeline(preprocessing_steps + [('model', model)])
                
                # Entraîner
                pipeline.fit(self.X_train, self.y_train)
                
                # Stocker
                self.model_pipelines[name] = pipeline
                
                print(f" {name} entraîné avec succès")
                
            except Exception as e:
                print(f" Erreur {name}: {str(e)}")
        
        print(f"\n {len(self.model_pipelines)} modèles entraînés avec succès!")
        return self.model_pipelines
    
    def evaluate_models(self, cv=5):
        """
        Évaluation des modèles avec cross-validation
        """
        print("\n ÉVALUATION DES MODÈLES")
        print("-" * 40)
        
        results = []
        scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        
        for name, pipeline in self.model_pipelines.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                          cv=cv, scoring=scoring, n_jobs=-1)
                
                # Prédictions
                y_train_pred = pipeline.predict(self.X_train)
                y_test_pred = pipeline.predict(self.X_test)
                
                if self.task_type == 'regression':
                    train_r2 = r2_score(self.y_train, y_train_pred)
                    test_r2 = r2_score(self.y_test, y_test_pred)
                    rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
                    
                    results.append({
                        'Model': name,
                        'CV_Score_Mean': cv_scores.mean(),
                        'CV_Score_Std': cv_scores.std(),
                        'Train_R2': train_r2,
                        'Test_R2': test_r2,
                        'RMSE': rmse,
                        'Overfitting': train_r2 - test_r2
                    })
                    
                    print(f"{name:15} | CV R²: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test R²: {test_r2:.3f} | RMSE: {rmse:.3f}")
                    
                else:
                    train_acc = accuracy_score(self.y_train, y_train_pred)
                    test_acc = accuracy_score(self.y_test, y_test_pred)
                    
                    results.append({
                        'Model': name,
                        'CV_Score_Mean': cv_scores.mean(),
                        'CV_Score_Std': cv_scores.std(),
                        'Train_Accuracy': train_acc,
                        'Test_Accuracy': test_acc,
                        'Overfitting': train_acc - test_acc
                    })
                    
                    print(f"{name:15} | CV Acc: {cv_scores.mean():.3f}±{cv_scores.std():.3f} | Test Acc: {test_acc:.3f}")
                
            except Exception as e:
                print(f" Erreur évaluation {name}: {str(e)}")
        
        results_df = pd.DataFrame(results)
        self.results['initial'] = results_df
        
        # Visualisation
        self._plot_model_comparison(results_df)
        
        return results_df
    
    
    def tune_hyperparameters(self, cv=5, n_jobs=-1, verbose=1):
        """
        5. Tuning des hyperparamètres avec GridSearchCV
        """
        print("="*60)
        print(" 5. TUNING DES HYPERPARAMÈTRES")
        print("="*60)
        
        scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        
        # Définir les grilles de paramètres pour chaque pipeline
        param_grids = self._get_param_grids()
        
        print(f" Configuration tuning:")
        print(f"   Cross-validation: {cv} folds")
        print(f"   Métrique: {scoring}")
        print(f"   Jobs parallèles: {n_jobs}")
        
        tuning_results = []
        
        for name, pipeline in self.model_pipelines.items():
            if name in param_grids:
                print(f"\n Tuning {name}...")
                
                try:
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grids[name],
                        cv=cv,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        verbose=verbose
                    )
                    
                    grid_search.fit(self.X_train, self.y_train)
                    
                    # Sauvegarder le meilleur pipeline
                    self.best_pipelines[name] = grid_search.best_estimator_
                    
                    best_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                    
                    # Calculer l'amélioration
                    initial_score = self.results['initial'][self.results['initial']['Model'] == name]['CV_Score_Mean'].iloc[0]
                    improvement = best_score - initial_score
                    
                    tuning_results.append({
                        'Model': name,
                        'Best_CV_Score': best_score,
                        'Improvement': improvement,
                        'Best_Params': str(best_params)
                    })
                    
                    print(f" Meilleur CV score: {best_score:.3f}")
                    print(f" Amélioration: {improvement:+.3f}")
                    
                except Exception as e:
                    print(f" Erreur: {str(e)}")
                    self.best_pipelines[name] = pipeline  
        
        tuning_df = pd.DataFrame(tuning_results)
        self.results['tuning'] = tuning_df
        
        print(f"\n Tuning terminé pour {len(self.best_pipelines)} modèles!")
        return tuning_df

    
    def evaluate_final_models(self):
        """Évaluation finale des meilleurs modèles"""
        print("\n ÉVALUATION FINALE DES MEILLEURS MODÈLES")
        print("-" * 50)
        
        final_results = []
        
        for name, pipeline in self.best_pipelines.items():
            try:
                y_train_pred = pipeline.predict(self.X_train)
                y_test_pred = pipeline.predict(self.X_test)
                
                if self.task_type == 'regression':
                    train_r2 = r2_score(self.y_train, y_train_pred)
                    test_r2 = r2_score(self.y_test, y_test_pred)
                    rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
                    
                    final_results.append({
                        'Model': name,
                        'Train_R2': train_r2,
                        'Test_R2': test_r2,
                        'RMSE': rmse,
                        'Overfitting': train_r2 - test_r2
                    })
                    
                    print(f"{name:15} | R² Test: {test_r2:.3f} | RMSE: {rmse:.3f} | Overfit: {train_r2 - test_r2:+.3f}")
                    
                else:
                    train_acc = accuracy_score(self.y_train, y_train_pred)
                    test_acc = accuracy_score(self.y_test, y_test_pred)
                    
                    final_results.append({
                        'Model': name,
                        'Train_Accuracy': train_acc,
                        'Test_Accuracy': test_acc,
                        'Overfitting': train_acc - test_acc
                    })
                    
                    print(f"{name:15} | Acc Test: {test_acc:.3f} | Overfit: {train_acc - test_acc:+.3f}")
                
            except Exception as e:
                print(f" Erreur {name}: {str(e)}")
        
        final_df = pd.DataFrame(final_results)
        self.results['final'] = final_df
        return final_df
    
    def select_best_features(self, top_k=10):
        """
        6. Sélection des variables basée sur l'importance
        """
        print("="*60)
        print(" 6. SÉLECTION DES VARIABLES")
        print("="*60)
        
        for model_name in ['Random_Forest', 'XGBoost']:
            if model_name in self.best_pipelines:
                pipeline = self.best_pipelines[model_name]
                model = pipeline.named_steps['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    n_features = len(importances)
                    if hasattr(pipeline.named_steps.get('pca'), 'n_components_'):
                        feature_names = [f'PC_{i+1}' for i in range(n_features)]
                    else:
                        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    self.feature_importance[model_name] = importance_df
                    
                    print(f"\n🌲 IMPORTANCES {model_name.upper()}:")
                    print("-" * 30)
                    print(importance_df.head(top_k).to_string(index=False))
        
        # Visualisation
        self._plot_feature_importance(top_k)
        
        return self.feature_importance
    
    def generate_final_report(self):
        """
        7. Rapport final et recommandations
        """
        print("="*60)
        print(" 7. RAPPORT FINAL ET RECOMMANDATIONS")
        print("="*60)
        
        if 'final' not in self.results:
            print(" Aucun résultat final à analyser!")
            return
        
        final_df = self.results['final']
        
        # Identifier le meilleur modèle
        if self.task_type == 'regression':
            best_model = final_df.loc[final_df['Test_R2'].idxmax()]
            metric = 'R²'
            score = best_model['Test_R2']
        else:
            best_model = final_df.loc[final_df['Test_Accuracy'].idxmax()]
            metric = 'Accuracy'
            score = best_model['Test_Accuracy']
        
        print(f" MEILLEUR MODÈLE:")
        print(f"   {best_model['Model']} - {metric}: {score:.3f}")
        print(f"   Overfitting: {best_model['Overfitting']:+.3f}")
        
        # Analyse du surapprentissage
        print(f"\n ANALYSE DU SURAPPRENTISSAGE:")
        for _, row in final_df.iterrows():
            overfitting = row['Overfitting']
            status = " Bon" if overfitting < 0.05 else " Modéré" if overfitting < 0.1 else " Élevé"
            print(f"   {row['Model']:15s} | {overfitting:+.3f} | {status}")
        
        # Recommandations
        print(f"\n RECOMMANDATIONS:")
        
        if best_model['Overfitting'] > 0.1:
            print(" Réduire le surapprentissage:")
            print("      - Augmenter la régularisation")
            print("      - Collecter plus de données")
        
        if self.task_type == 'regression' and score < 0.8:
            print(" Améliorer les performances:")
            print("      - Feature engineering avancé")
            print("      - Essayer ensemble methods")
        
        print(f"\n FEATURES IMPORTANTES:")
        if self.feature_importance:
            for model_name, importance_df in self.feature_importance.items():
                top_feature = importance_df.iloc[0]
                print(f"   {model_name}: {top_feature['Feature']} ({top_feature['Importance']:.3f})")
    
        self._plot_final_summary()


    """
        HELPER METHODS

    """    

    def _plot_pca_analysis(self, explained_variance, cumulative_variance, X_train_pca):
        """Visualisations pour l'analyse PCA"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Variance expliquée
        axes[0,0].bar(range(1, len(explained_variance)+1), explained_variance)
        axes[0,0].set_title('Variance Expliquée par Composante')
        axes[0,0].set_xlabel('Composante Principale')
        axes[0,0].set_ylabel('Variance Expliquée')
        
        # 2. Variance cumulée
        axes[0,1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
        axes[0,1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        axes[0,1].set_title('Variance Cumulée')
        axes[0,1].set_xlabel('Nombre de Composantes')
        axes[0,1].set_ylabel('Variance Cumulée')
        axes[0,1].legend()
        
        # 3. Biplot (si au moins 2 composantes)
        if X_train_pca.shape[1] >= 2:
            axes[1,0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.6)
            axes[1,0].set_title('Biplot PC1 vs PC2')
            axes[1,0].set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
            axes[1,0].set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
        
        # 4. Distribution PC1
        axes[1,1].hist(X_train_pca[:, 0], bins=30, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Distribution de PC1')
        axes[1,1].set_xlabel('PC1')
        axes[1,1].set_ylabel('Fréquence')
        
        plt.tight_layout()
        plt.show()


    def _plot_model_comparison(self, results_df):
        """Visualisation de la comparaison des modèles"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if self.task_type == 'regression':
            # R² scores
            x = np.arange(len(results_df))
            width = 0.35
            axes[0].bar(x - width/2, results_df['Train_R2'], width, label='Train R²', alpha=0.8)
            axes[0].bar(x + width/2, results_df['Test_R2'], width, label='Test R²', alpha=0.8)
            axes[0].set_xlabel('Modèles')
            axes[0].set_ylabel('R² Score')
            axes[0].set_title('Comparaison R² Train vs Test')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(results_df['Model'], rotation=45)
            axes[0].legend()
            
            # RMSE
            axes[1].bar(results_df['Model'], results_df['RMSE'], alpha=0.8, color='orange')
            axes[1].set_xlabel('Modèles')
            axes[1].set_ylabel('RMSE')
            axes[1].set_title('RMSE par Modèle')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            # Accuracy
            x = np.arange(len(results_df))
            width = 0.35
            axes[0].bar(x - width/2, results_df['Train_Accuracy'], width, label='Train Accuracy', alpha=0.8)
            axes[0].bar(x + width/2, results_df['Test_Accuracy'], width, label='Test Accuracy', alpha=0.8)
            axes[0].set_xlabel('Modèles')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Comparaison Accuracy Train vs Test')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(results_df['Model'], rotation=45)
            axes[0].legend()
            
            # Overfitting
            axes[1].bar(results_df['Model'], results_df['Overfitting'], alpha=0.8, color='red')
            axes[1].set_xlabel('Modèles')
            axes[1].set_ylabel('Overfitting')
            axes[1].set_title('Analyse du Surapprentissage')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


    def _get_param_grids(self):
        """Définir les grilles de paramètres pour chaque modèle"""
        param_grids = {
            'SVM': {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto', 0.001, 0.01],
                'model__kernel': ['rbf', 'poly']
            },
            'Decision_Tree': {
                'model__max_depth': [3, 5, 7, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'Random_Forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7, None],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            },
            'XGBoost': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        return param_grids


    def _plot_feature_importance(self, top_k):
        """Visualisation de l'importance des features"""
        if not self.feature_importance:
            return
        
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(self.feature_importance.items()):
            top_features = importance_df.head(top_k)
            axes[i].barh(top_features['Feature'], top_features['Importance'])
            axes[i].set_title(f'Feature Importance - {model_name}')
            axes[i].set_xlabel('Importance')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
            
    
    def _plot_final_summary(self):
        """Visualisation du résumé final"""
        if 'final' not in self.results:
            return
        
        final_df = self.results['final']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance finale
        if self.task_type == 'regression':
            axes[0,0].bar(final_df['Model'], final_df['Test_R2'], alpha=0.8)
            axes[0,0].set_title('Performance R² Finale')
            axes[0,0].set_ylabel('R² Test')
            axes[0,0].tick_params(axis='x', rotation=45)
        else:
            axes[0,0].bar(final_df['Model'], final_df['Test_Accuracy'], alpha=0.8)
            axes[0,0].set_title('Performance Accuracy Finale')
            axes[0,0].set_ylabel('Accuracy Test')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Overfitting
        colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in final_df['Overfitting']]
        axes[0,1].bar(final_df['Model'], final_df['Overfitting'], color=colors, alpha=0.8)
        axes[0,1].set_title('Analyse du Surapprentissage')
        axes[0,1].set_ylabel('Overfitting (Train - Test)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7)
        axes[0,1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
        
        # Amélioration due au tuning
        if 'tuning' in self.results:
            tuning_df = self.results['tuning']
            axes[1,0].bar(tuning_df['Model'], tuning_df['Improvement'], alpha=0.8, color='purple')
            axes[1,0].set_title('Amélioration due au Tuning')
            axes[1,0].set_ylabel('Amélioration Score')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Pipeline components used
        components = ['Preprocessing', 'Scaling', 'Model']
        if self.pca_pipeline is not None:
            components.insert(-1, 'PCA')
        
        axes[1,1].pie([1]*len(components), labels=components, autopct='%1.0f%%')
        axes[1,1].set_title('Composants du Pipeline')
        
        plt.tight_layout()
        plt.show()