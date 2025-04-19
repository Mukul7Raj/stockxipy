# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from llm_model import predict_stock_price
import plotly.utils
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            symbol = request.form.get('symbol', '').strip().upper()
            
            if not symbol:
                return render_template('index.html', error="Please enter a stock symbol")

            # Get predictions and analysis
            result = predict_stock_price(symbol)
            
            # Convert plot to JSON
            plot_json = json.dumps(result['plot'], cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template('index.html',
                                symbol=symbol,
                                plot=plot_json,
                                prediction=result)
                                
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True) 