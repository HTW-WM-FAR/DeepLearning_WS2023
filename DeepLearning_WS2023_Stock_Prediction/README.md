# stockPred-Mex
Stock prediction (S&amp;P 100) with Random Forest and XGBoost algortihms

Simulate trading based on stock prediction models using CSV files for multiple stocks.

Key functionalities include:

Reading multiple CSV files containing stock prediction data for different stocks.
A function to simulate trading for a specific stock based on predictions (positive, negative, or flat).
Accumulating the final values of each stock's simulated trading into a global sum.
Prompting users to input the number of days they want to simulate for all stocks.

# Instructions
1. $ git clone <Project A>  # Cloning project repository
2. $ cd <Project A> # Enter to project directory
3. $ python3 -m virtualenv venv  
4. source venv/bin/activate
5. (my_venv)$ python3 -m pip install -r ./requirements.txt  
6. (my_venv)$ deactivate # When you want to leave virtual environment

7. Under folder *lstm* run lstm.ipynb and simulation.py
9. Under folder *randomForest* run randomForest.py and simulation.py
10. Under folder *xgboost* Run tsf_xgboost.ipynb and simulation.py
