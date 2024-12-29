from flask import Flask, request, jsonify  
import numpy as np  
import joblib  
from flask_cors import CORS  

app = Flask(__name__)  
CORS(app)  # Kích hoạt CORS  

# Tải mô hình đã huấn luyện  
model = joblib.load(r'C:\Users\ASUS\Documents\housing_price\best_model.joblib')  # Đường dẫn đến mô hình của bạn  

@app.route('/predict', methods=['POST'])  
def predict():  
    data = request.json  

    # Chuyển đổi dữ liệu đầu vào thành mảng numpy cho mô hình  
    features = np.array([[data['tradeTime'],  
                          data['followers'],  
                          data['square'],  
                          data['drawingRoom'],  
                          data['constructionTime'],  
                          data['renovationCondition'],  
                          data['subway'],  
                          data['communityAverage'],  
                          data['distance'],  
                          data['Age']]])  

    # Dự đoán  
    prediction = model.predict(features)  

    # Kiểm tra giá trị dự đoán  
    predicted_price = prediction[0]  

    # Đảm bảo nó là kiểu số thực  
    if isinstance(predicted_price, np.float32) or isinstance(predicted_price, np.float64):  
        predicted_price = float(predicted_price)  

    return jsonify({'totalPrice': predicted_price})  

if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0', port=5000)