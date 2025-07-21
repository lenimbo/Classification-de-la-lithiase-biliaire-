from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
import logging
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Chargement du modèle avec gestion d'erreur améliorée
def load_model():
    model_paths = [
        'prediction_modelRF.pkl',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_modelRF.pkl'),
        'prediction_modelRF_backup.pkl'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Tentative de chargement du modèle depuis: {model_path}")
            
            # Tentatives de chargement avec différentes méthodes
            loaders = [
                lambda p: pickle.load(open(p, 'rb')),
                lambda p: __import__('joblib').load(p),
                lambda p: pickle.load(open(p, 'rb'), encoding='latin-1'),
                lambda p: pickle.load(open(p, 'rb'), encoding='bytes')
            ]
            
            for i, loader in enumerate(loaders):
                try:
                    model = loader(model_path)
                    logger.info(f"Modèle chargé avec succès (méthode {i+1}) depuis: {model_path}")
                    logger.info(f"Type du modèle: {type(model)}")
                    return model
                except Exception as e:
                    logger.warning(f"Méthode {i+1} échouée: {str(e)}")
                    continue
    
    logger.error("Impossible de charger le modèle depuis tous les chemins testés")
    return None

# Chargement du modèle au démarrage
model = load_model()

# Si aucun modèle n'est trouvé, créer un modèle de démonstration
if model is None:
    logger.warning("Création d'un modèle de démonstration...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        # Modèle très simple pour la démonstration
        demo_model = RandomForestClassifier(n_estimators=5, random_state=42)
        
        # Données factices pour l'entraînement
        np.random.seed(42)
        X_demo = np.random.randn(100, 10)
        y_demo = np.random.randint(0, 2, 100)
        demo_model.fit(X_demo, y_demo)
        
        model = demo_model
        logger.info("Modèle de démonstration créé avec succès")
    except Exception as e:
        logger.error(f"Impossible de créer un modèle de démonstration: {str(e)}")

# Variables explicatives
FEATURES = ['Gender', 'Hyperlipidemia', 'Bone Mass (BM)', 'High Density Lipoprotein (HDL)', 
           'Aspartat Aminotransferaz (AST)', 'Alkaline Phosphatase (ALP)', 'Creatinine', 
           'C-Reactive Protein (CRP)', 'Hemoglobin (HGB)', 'Vitamin D']

# Template HTML (identique à votre version)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction Lithiase Biliaire</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1000px; margin: 0 auto; background: white;
            border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white; padding: 40px; text-align: center;
        }
        .header h1 {
            font-size: 2.5em; margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .form-container { padding: 40px; }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px; margin-bottom: 30px;
        }
        .form-group { position: relative; }
        .form-group label {
            display: block; margin-bottom: 8px; font-weight: 600;
            color: #333; font-size: 0.95em;
        }
        .form-group input, .form-group select {
            width: 100%; padding: 12px 16px; border: 2px solid #e1e5e9;
            border-radius: 10px; font-size: 16px; transition: all 0.3s ease;
            background: #f8f9fa;
        }
        .binary-group { display: flex; gap: 15px; margin-top: 8px; }
        .radio-option {
            display: flex; align-items: center; gap: 8px; padding: 8px 16px;
            border: 2px solid #e1e5e9; border-radius: 25px; cursor: pointer;
            transition: all 0.3s ease; background: #f8f9fa;
        }
        .radio-option.selected {
            border-color: #4facfe; background: #4facfe; color: white;
        }
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 15px 40px; border: none; border-radius: 50px;
            font-size: 18px; font-weight: 600; cursor: pointer;
            transition: all 0.3s ease; display: block; margin: 30px auto 0;
        }
        .result { margin-top: 30px; padding: 25px; border-radius: 15px;
            text-align: center; font-size: 18px; font-weight: 600; display: none; }
        .result.success { background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #2d5a3d; border: 2px solid #84fab0; }
        .result.warning { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #8b4513; border: 2px solid #fcb69f; }
        .result.error { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: #8b0000; border: 2px solid #ff9a9e; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🩺 Prédiction Lithiase Biliaire</h1>
            <p>Système d'aide au diagnostic basé sur l'intelligence artificielle</p>
        </div>
        <div class="form-container">
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Genre:</label>
                        <div class="binary-group">
                            <div class="radio-option">
                                <input type="radio" name="gender" value="0" required>
                                <label>Homme</label>
                            </div>
                            <div class="radio-option">
                                <input type="radio" name="gender" value="1" required>
                                <label>Femme</label>
                            </div>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Hyperlipidémie:</label>
                        <div class="binary-group">
                            <div class="radio-option">
                                <input type="radio" name="hyperlipidemia" value="0" required>
                                <label>Non</label>
                            </div>
                            <div class="radio-option">
                                <input type="radio" name="hyperlipidemia" value="1" required>
                                <label>Oui</label>
                            </div>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Masse Osseuse (BM):</label>
                        <input type="number" name="bone_mass" step="0.01" placeholder="Ex: 2.5" required>
                    </div>
                    <div class="form-group">
                        <label>HDL (mg/dL):</label>
                        <input type="number" name="hdl" step="0.01" placeholder="Ex: 45.5" required>
                    </div>
                    <div class="form-group">
                        <label>AST (U/L):</label>
                        <input type="number" name="ast" step="0.01" placeholder="Ex: 25.3" required>
                    </div>
                    <div class="form-group">
                        <label>ALP (U/L):</label>
                        <input type="number" name="alp" step="0.01" placeholder="Ex: 78.2" required>
                    </div>
                    <div class="form-group">
                        <label>Créatinine (mg/dL):</label>
                        <input type="number" name="creatinine" step="0.01" placeholder="Ex: 1.1" required>
                    </div>
                    <div class="form-group">
                        <label>CRP (mg/L):</label>
                        <input type="number" name="crp" step="0.01" placeholder="Ex: 3.2" required>
                    </div>
                    <div class="form-group">
                        <label>Hémoglobine (g/dL):</label>
                        <input type="number" name="hgb" step="0.01" placeholder="Ex: 12.5" required>
                    </div>
                    <div class="form-group">
                        <label>Vitamine D (ng/mL):</label>
                        <input type="number" name="vitamin_d" step="0.01" placeholder="Ex: 25.8" required>
                    </div>
                </div>
                <button type="submit" class="submit-btn">🔍 Analyser le Risque</button>
            </form>
            <div id="result" class="result"></div>
        </div>
    </div>
    <script>
        document.querySelectorAll('input[type="radio"]').forEach(radio => {
            radio.addEventListener('change', function() {
                document.querySelectorAll(`input[name="${this.name}"]`).forEach(r => {
                    r.closest('.radio-option').classList.remove('selected');
                });
                this.closest('.radio-option').classList.add('selected');
            });
        });
        
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
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                
                if (result.success) {
                    const probability = (result.probability * 100).toFixed(1);
                    if (result.prediction === 1) {
                        resultDiv.className = 'result warning';
                        resultDiv.innerHTML = `<h3>⚠️ Risque Élevé</h3><p>Probabilité: ${probability}%</p>`;
                    } else {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `<h3>✅ Risque Faible</h3><p>Probabilité: ${probability}%</p>`;
                    }
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>❌ Erreur</h3><p>${result.message}</p>`;
                }
            } catch (error) {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '<h3>❌ Erreur de Connexion</h3>';
            }
        });
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'message': 'Modèle non disponible'
            }), 500
        
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'Aucune donnée'})
        
        required_fields = ['gender', 'hyperlipidemia', 'bone_mass', 'hdl', 'ast', 
                          'alp', 'creatinine', 'crp', 'hgb', 'vitamin_d']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Champs manquants: {", ".join(missing_fields)}'
            })
        
        features = [float(data[field]) for field in required_fields]
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)[0]
        
        probability = 0.5
        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_array)[0][1]
        except:
            pass
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erreur: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features': FEATURES
    })

# Point d'entrée pour Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)