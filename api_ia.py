import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    modelo_regressao = joblib.load('modelo_tempo_ciclo.joblib')
    modelo_classificacao = joblib.load('modelo_gargalo.joblib')
    print("Modelos de IA carregados com sucesso.")
except FileNotFoundError:
    print("ERRO: Arquivos .joblib não encontrados. Treine os modelos primeiro.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe dados de um layout e retorna a predição de tempo e gargalo.
    Formato esperado do JSON:
    {
        "total_maquinas": 3,
        "soma_tempos_ciclo": 462.5
    }
    """
    try:
        data = request.get_json()

        total_maquinas = data['total_maquinas']
        soma_tempos_ciclo = data['soma_tempos_ciclo']

        input_features = [[total_maquinas, soma_tempos_ciclo]]

        tempo_previsto = modelo_regressao.predict(input_features)[0]
        gargalo_id = modelo_classificacao.predict(input_features)[0]

        return jsonify({
            'tempo_ciclo_total': round(tempo_previsto, 2),
            'maquina_gargalo_id': int(gargalo_id)
        })

    except Exception as e:
        return jsonify({'erro': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)