import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from tradingview_ta import TA_Handler, Interval

gold_handler = TA_Handler(
    symbol="XAUUSD",
    exchange="OANDA",
    screener="cfd",
    interval=Interval.INTERVAL_1_HOUR_MINUTE,
    timeout=10
)

# Fetch the technical analysis data
gold_data = gold_handler.get_analysis()

# Input variables
rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
adx = ctrl.Antecedent(np.arange(0, 101, 1), 'adx')
CCI20 = ctrl.Antecedent(np.arange(-300, 301, 1), 'CCI20')

# Output variable
decision = ctrl.Consequent(np.arange(0, 101, 1), 'decision')

# Membership functions for input variables
rsi['low'] = fuzz.trimf(rsi.universe, [0, 30, 60])
rsi['medium'] = fuzz.trimf(rsi.universe, [30, 60, 90])
rsi['high'] = fuzz.trimf(rsi.universe, [60, 90, 100])

adx['weak_trend'] = fuzz.trimf(adx.universe, [0, 10, 25])
adx['moderate_trend'] = fuzz.trimf(adx.universe, [25, 35, 50])
adx['strong_trend'] = fuzz.trimf(adx.universe, [50, 60, 75])
adx['very_strong_trend'] = fuzz.trimf(adx.universe, [75, 90, 100])

CCI20['over_sold'] = fuzz.trimf(CCI20.universe, [-300, -200, -100])
CCI20['normal'] = fuzz.trimf(CCI20.universe, [-100, 0, 100])
CCI20['over_bought'] = fuzz.trimf(CCI20.universe, [100, 200, 300])

# Membership functions for output variable
decision['sell'] = fuzz.trimf(decision.universe, [0, 30, 60])
decision['hold'] = fuzz.trimf(decision.universe, [30, 60, 90])
decision['buy'] = fuzz.trimf(decision.universe, [60, 90, 100])

# Rules
rule1 = ctrl.Rule(rsi['low'] & adx['weak_trend'] & CCI20['over_sold'], decision['buy'])
rule2 = ctrl.Rule(rsi['low'] & adx['weak_trend'] & CCI20['normal'], decision['hold'])
rule3 = ctrl.Rule(rsi['low'] & adx['weak_trend'] & CCI20['over_bought'], decision['sell'])

rule4 = ctrl.Rule(rsi['medium'] & adx['moderate_trend'] & CCI20['over_sold'], decision['buy'])
rule5 = ctrl.Rule(rsi['medium'] & adx['moderate_trend'] & CCI20['normal'], decision['hold'])
rule6 = ctrl.Rule(rsi['medium'] & adx['moderate_trend'] & CCI20['over_bought'], decision['sell'])

rule7 = ctrl.Rule(rsi['high'] & adx['strong_trend'] & CCI20['over_sold'], decision['buy'])
rule8 = ctrl.Rule(rsi['high'] & adx['strong_trend'] & CCI20['normal'], decision['hold'])
rule9 = ctrl.Rule(rsi['high'] & adx['strong_trend'] & CCI20['over_bought'], decision['sell'])

rule10 = ctrl.Rule(rsi['high'] & adx['very_strong_trend'] & CCI20['over_sold'], decision['buy'])
rule11 = ctrl.Rule(rsi['high'] & adx['very_strong_trend'] & CCI20['normal'], decision['hold'])
rule12 = ctrl.Rule(rsi['high'] & adx['very_strong_trend'] & CCI20['over_bought'], decision['sell'])

# Control system
decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
decision_simulation = ctrl.ControlSystemSimulation(decision_ctrl)

# Set inputs
decision_simulation.input['rsi'] = gold_data.indicators.get("RSI")
decision_simulation.input['adx'] = gold_data.indicators.get("ADX")
decision_simulation.input['CCI20'] = gold_data.indicators.get("CCI20")

# Compute the decision
decision_simulation.compute()

# Print the decision
print("Decision:", decision_simulation.output['decision'])

# Determine the linguistic label of the decision
decision_label = None
if decision_simulation.output['decision'] <= 30:
    decision_label = "sell"
elif decision_simulation.output['decision'] <= 60:
    decision_label = "hold"
else:
    decision_label = "buy"

# Print the linguistic label of the decision
print("Decision Label:", decision_label)

# Visualize the membership functions and decision
rsi.view()
adx.view()
CCI20.view()
decision.view()
plt.show()