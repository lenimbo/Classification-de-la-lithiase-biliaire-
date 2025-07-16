from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Chargement du modèle
import os
import joblib
model_path = 'prediction_modelRF.pkl'
# Si le fichier n'est pas trouvé dans le répertoire courant, chercher dans le répertoire du script
if not os.path.exists(model_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'prediction_modelRF.pkl')

model = None
try:
    # Première tentative avec pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Modèle chargé avec succès (pickle) depuis: {model_path}")
    logger.info(f"Type du modèle: {type(model)}")
except Exception as e:
    logger.warning(f"Erreur avec pickle: {str(e)}")
    try:
        # Deuxième tentative avec joblib
        model = joblib.load(model_path)
        logger.info(f"Modèle chargé avec succès (joblib) depuis: {model_path}")
        logger.info(f"Type du modèle: {type(model)}")
    except Exception as e2:
        logger.error(f"Erreur avec joblib: {str(e2)}")
        try:
            # Troisième tentative avec pickle en mode latin-1 (pour compatibilité Python 2/3)
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding='latin-1')
            logger.info(f"Modèle chargé avec succès (pickle latin-1) depuis: {model_path}")
            logger.info(f"Type du modèle: {type(model)}")
        except Exception as e3:
            logger.error(f"Erreur avec pickle latin-1: {str(e3)}")
            try:
                # Quatrième tentative avec pickle en mode bytes
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='bytes')
                logger.info(f"Modèle chargé avec succès (pickle bytes) depuis: {model_path}")
                logger.info(f"Type du modèle: {type(model)}")
            except Exception as e4:
                logger.error(f"Toutes les tentatives de chargement ont échoué:")
                logger.error(f"  - Pickle: {str(e)}")
                logger.error(f"  - Joblib: {str(e2)}")
                logger.error(f"  - Pickle latin-1: {str(e3)}")
                logger.error(f"  - Pickle bytes: {str(e4)}")
                logger.error(f"Répertoire courant: {os.getcwd()}")
                logger.error(f"Répertoire du script: {os.path.dirname(os.path.abspath(__file__))}")
                model = None

if model is not None:
    logger.info("=== INFORMATIONS SUR LE MODÈLE ===")
    logger.info(f"Type: {type(model)}")
    logger.info(f"Méthodes disponibles: {[m for m in dir(model) if not m.startswith('_')]}")
    
    # Test rapide avec des données factices
    try:
        test_data = [[1, 0, 2.5, 45.5, 25.3, 78.2, 1.1, 3.2, 12.5, 25.8]]
        test_prediction = model.predict(test_data)
        logger.info(f"Test de prédiction réussi: {test_prediction}")
    except Exception as e:
        logger.error(f"Erreur lors du test de prédiction: {str(e)}")
else:
    logger.error("AUCUN MODÈLE N'A PU ÊTRE CHARGÉ - L'APPLICATION FONCTIONNERA EN MODE DÉGRADÉ")

# Variables explicatives
FEATURES = ['Gender', 'Hyperlipidemia', 'Bone Mass (BM)', 'High Density Lipoprotein (HDL)', 
           'Aspartat Aminotransferaz (AST)', 'Alkaline Phosphatase (ALP)', 'Creatinine', 
           'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D']

# Template HTML pour l'interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction Lithiase Biliaire</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .form-container {
            padding: 40px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .form-group {
            position: relative;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #4facfe;
            background: white;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
        }
        
        .binary-group {
            display: flex;
            gap: 15px;
            margin-top: 8px;
        }
        
        .radio-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        
        .radio-option:hover {
            border-color: #4facfe;
            background: white;
        }
        
        .radio-option input[type="radio"] {
            width: auto;
            margin: 0;
        }
        
        .radio-option.selected {
            border-color: #4facfe;
            background: #4facfe;
            color: white;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            margin: 30px auto 0;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            display: none;
        }
        
        .result.success {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #2d5a3d;
            border: 2px solid #84fab0;
        }
        
        .result.warning {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #8b4513;
            border: 2px solid #fcb69f;
        }
        
        .result.error {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #8b0000;
            border: 2px solid #ff9a9e;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .info-box {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            border-left: 5px solid #4facfe;
        }
        
        .info-box h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .info-box p {
            color: #666;
            line-height: 1.6;
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .form-container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩺 Prédiction Lithiase Biliaire</h1>
            <p>Système d'aide au diagnostic basé sur l'intelligence artificielle</p>
        </div>
        
        <div class="form-container">
            <div class="info-box">
                <h3>ℹ️ Informations importantes</h3>
                <p>Cet outil utilise un modèle d'intelligence artificielle pour prédire le risque de lithiase biliaire basé sur des données cliniques. 
                Cette prédiction ne remplace pas un diagnostic médical professionnel.</p>
            </div>
            
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="gender">Genre:</label>
                        <div class="binary-group">
                            <div class="radio-option">
                                <input type="radio" id="male" name="gender" value="0" required>
                                <label for="male">Homme</label>
                            </div>
                            <div class="radio-option">
                                <input type="radio" id="female" name="gender" value="1" required>
                                <label for="female">Femme</label>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="hyperlipidemia">Hyperlipidémie:</label>
                        <div class="binary-group">
                            <div class="radio-option">
                                <input type="radio" id="hyper_no" name="hyperlipidemia" value="0" required>
                                <label for="hyper_no">Non</label>
                            </div>
                            <div class="radio-option">
                                <input type="radio" id="hyper_yes" name="hyperlipidemia" value="1" required>
                                <label for="hyper_yes">Oui</label>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="bone_mass">Masse Osseuse (BM):</label>
                        <input type="number" id="bone_mass" name="bone_mass" step="0.01" placeholder="Ex: 2.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="hdl">Lipoprotéine Haute Densité (HDL) mg/dL:</label>
                        <input type="number" id="hdl" name="hdl" step="0.01" placeholder="Ex: 45.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="ast">Aspartate Aminotransférase (AST) U/L:</label>
                        <input type="number" id="ast" name="ast" step="0.01" placeholder="Ex: 25.3" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="alp">Phosphatase Alcaline (ALP) U/L:</label>
                        <input type="number" id="alp" name="alp" step="0.01" placeholder="Ex: 78.2" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="creatinine">Créatinine mg/dL:</label>
                        <input type="number" id="creatinine" name="creatinine" step="0.01" placeholder="Ex: 1.1" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="crp">Protéine C-Réactive (CRP) mg/L:</label>
                        <input type="number" id="crp" name="crp" step="0.01" placeholder="Ex: 3.2" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="hgb">Hémoglobine (HGB) g/dL:</label>
                        <input type="number" id="hgb" name="hgb" step="0.01" placeholder="Ex: 12.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="vitamin_d">Vitamine D ng/mL:</label>
                        <input type="number" id="vitamin_d" name="vitamin_d" step="0.01" placeholder="Ex: 25.8" required>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn">🔍 Analyser le Risque</button>
            </form>
            
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyse en cours...</p>
            </div>
            
            <div id="result" class="result"></div>
        </div>
    </div>

    <script>
        // Gestion des boutons radio stylisés
        document.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener('change', function() {
                // Retirer la classe selected de tous les options du même groupe
                document.querySelectorAll(`input[name="${this.name}"]`).forEach(r => {
                    r.closest('.radio-option').classList.remove('selected');
                });
                // Ajouter la classe selected à l'option sélectionnée
                this.closest('.radio-option').classList.add('selected');
            });
        });
        
        // Gestion du formulaire
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {
                gender: parseFloat(formData.get('gender')),
                hyperlipidemia: parseFloat(formData.get('hyperlipidemia')),
                bone_mass: parseFloat(formData.get('bone_mass')),
                hdl: parseFloat(formData.get('hdl')),
                ast: parseFloat(formData.get('ast')),
                alp: parseFloat(formData.get('alp')),
                creatinine: parseFloat(formData.get('creatinine')),
                crp: parseFloat(formData.get('crp')),
                hgb: parseFloat(formData.get('hgb')),
                vitamin_d: parseFloat(formData.get('vitamin_d'))
            };
            
            // Afficher le loading
            document.querySelector('.loading').style.display = 'block';
            document.querySelector('.submit-btn').disabled = true;
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Cacher le loading
                document.querySelector('.loading').style.display = 'none';
                document.querySelector('.submit-btn').disabled = false;
                
                // Afficher le résultat
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (result.success) {
                    const probability = (result.probability * 100).toFixed(1);
                    if (result.prediction === 1) {
                        resultDiv.className = 'result warning';
                        resultDiv.innerHTML = `
                            <h3>⚠️ Risque Élevé Détecté</h3>
                            <p>Probabilité de lithiase biliaire: ${probability}%</p>
                            <p style="margin-top: 10px; font-size: 14px; font-weight: normal;">
                                Recommandation: Consulter un médecin pour un examen approfondi.
                            </p>
                        `;
                    } else {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3>✅ Risque Faible</h3>
                            <p>Probabilité de lithiase biliaire: ${probability}%</p>
                            <p style="margin-top: 10px; font-size: 14px; font-weight: normal;">
                                Les paramètres analysés suggèrent un risque faible.
                            </p>
                        `;
                    }
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `
                        <h3>❌ Erreur</h3>
                        <p>${result.message}</p>
                    `;
                }
            } catch (error) {
                // Cacher le loading
                document.querySelector('.loading').style.display = 'none';
                document.querySelector('.submit-btn').disabled = false;
                
                // Afficher l'erreur
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `
                    <h3>❌ Erreur de Connexion</h3>
                    <p>Impossible de se connecter au serveur. Veuillez réessayer.</p>
                `;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Page d'accueil avec le formulaire de prédiction"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint API pour la prédiction"""
    try:
        if model is None:
            logger.error("Modèle non chargé")
            return jsonify({
                'success': False,
                'message': 'Modèle non disponible. Vérifiez que le fichier prediction_modelRF.pkl existe.'
            }), 500
        
        # Récupération des données JSON
        data = request.json
        logger.info(f"Données reçues: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'Aucune donnée fournie'
            }), 400
        
        # Validation des données requises
        required_fields = ['gender', 'hyperlipidemia', 'bone_mass', 'hdl', 'ast', 
                          'alp', 'creatinine', 'crp', 'hgb', 'vitamin_d']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Champs manquants: {missing_fields}")
            return jsonify({
                'success': False,
                'message': f'Champs manquants: {", ".join(missing_fields)}'
            }), 400
        
        # Validation des valeurs numériques
        for field in required_fields:
            try:
                float(data[field])
            except (ValueError, TypeError):
                logger.error(f"Valeur invalide pour {field}: {data[field]}")
                return jsonify({
                    'success': False,
                    'message': f'Valeur invalide pour {field}: {data[field]}'
                }), 400
        
        # Préparation des données pour la prédiction
        # L'ordre doit correspondre exactement à celui des features du modèle
        features = [
            float(data['gender']),
            float(data['hyperlipidemia']),
            float(data['bone_mass']),
            float(data['hdl']),
            float(data['ast']),
            float(data['alp']),
            float(data['creatinine']),
            float(data['crp']),
            float(data['hgb']),
            float(data['vitamin_d'])
        ]
        
        logger.info(f"Features préparées: {features}")
        
        # Conversion en array numpy
        features_array = np.array(features).reshape(1, -1)
        logger.info(f"Shape du tableau: {features_array.shape}")
        
        # Vérification que le modèle a la méthode predict
        if not hasattr(model, 'predict'):
            logger.error("Le modèle n'a pas de méthode predict")
            return jsonify({
                'success': False,
                'message': 'Modèle invalide: méthode predict manquante'
            }), 500
        
        # Prédiction
        prediction = model.predict(features_array)[0]
        logger.info(f"Prédiction brute: {prediction}")
        
        # Probabilité (si le modèle le supporte)
        probability = 0.5  # Valeur par défaut
        try:
            if hasattr(model, 'predict_proba'):
                proba_result = model.predict_proba(features_array)
                logger.info(f"Probabilités brutes: {proba_result}")
                if proba_result.shape[1] > 1:
                    probability = proba_result[0][1]  # Probabilité de la classe positive
                else:
                    probability = proba_result[0][0]
            else:
                logger.info("Le modèle ne supporte pas predict_proba")
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de probabilité: {str(e)}")
        
        # Log de la prédiction
        logger.info(f"Prédiction finale: {prediction}, Probabilité: {probability:.3f}")
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'Prédiction réalisée avec succès'
        })
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.error(f"Type d'erreur: {type(e)}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Erreur lors de la prédiction: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Endpoint de vérification de l'état de l'application"""
    model_info = {}
    if model is not None:
        model_info = {
            'type': str(type(model)),
            'has_predict': hasattr(model, 'predict'),
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'attributes': [attr for attr in dir(model) if not attr.startswith('_')]
        }
        try:
            if hasattr(model, 'feature_names_in_'):
                model_info['feature_names'] = model.feature_names_in_.tolist()
            if hasattr(model, 'n_features_in_'):
                model_info['n_features'] = model.n_features_in_
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features': FEATURES,
        'model_info': model_info
    })

@app.route('/create_dummy_model')
def create_dummy_model():
    """Endpoint pour créer un modèle factice en cas de problème"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Création de données factices pour l'entraînement
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 10)  # 10 features
        y = np.random.randint(0, 2, n_samples)  # Classification binaire
        
        # Entraînement d'un modèle simple
        model_temp = RandomForestClassifier(n_estimators=10, random_state=42)
        model_temp.fit(X, y)
        
        # Sauvegarde
        import joblib
        joblib.dump(model_temp, 'prediction_modelRF_backup.pkl')
        
        return jsonify({
            'success': True,
            'message': 'Modèle factice créé avec succès',
            'file': 'prediction_modelRF_backup.pkl'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erreur lors de la création du modèle: {str(e)}'
        })

@app.route('/model_info')
def model_info():
    """Endpoint pour obtenir des informations détaillées sur le modèle"""
    if model is None:
        return jsonify({
            'model_loaded': False,
            'message': 'Aucun modèle chargé'
        })
    
    info = {
        'model_loaded': True,
        'model_type': str(type(model)),
        'model_methods': [m for m in dir(model) if not m.startswith('_')],
    }
    
    # Informations spécifiques aux modèles scikit-learn
    try:
        if hasattr(model, 'n_features_in_'):
            info['n_features_in'] = model.n_features_in_
        if hasattr(model, 'feature_names_in_'):
            info['feature_names_in'] = model.feature_names_in_.tolist()
        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
    except:
        pass
    
    return jsonify(info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint API alternatif pour intégration externe"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'code': 'MODEL_NOT_FOUND'
            }), 500
        
        data = request.json
        if not data:
            return jsonify({
                'error': 'No data provided',
                'code': 'NO_DATA'
            }), 400
        
        # Validation avec les noms exacts des features
        if 'features' not in data:
            return jsonify({
                'error': 'Features array required',
                'code': 'MISSING_FEATURES',
                'expected_format': {
                    'features': FEATURES
                }
            }), 400
        
        features = data['features']
        if len(features) != len(FEATURES):
            return jsonify({
                'error': f'Expected {len(FEATURES)} features, got {len(features)}',
                'code': 'INVALID_FEATURES_LENGTH',
                'expected_features': FEATURES
            }), 400
        
        # Prédiction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        try:
            probability = model.predict_proba(features_array)[0][1]
        except:
            probability = None
        
        response = {
            'prediction': int(prediction),
            'risk_level': 'high' if prediction == 1 else 'low'
        }
        
        if probability is not None:
            response['probability'] = float(probability)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'code': 'PREDICTION_ERROR'
        }), 500

if __name__ == '__main__':
    print("🚀 Démarrage de l'application de prédiction de lithiase biliaire")
    print("📊 Features utilisées:", FEATURES)
    print("🔗 Interface web: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/predict")
    print("🔗 Health check: http://localhost:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)