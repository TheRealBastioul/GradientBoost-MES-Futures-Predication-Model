from getPrediction import get_predictions
from showCharts import showCharts

ohyea = get_predictions('MES=F', 'NQ=F', 'JPY=X')
the_charts = showCharts(ohyea)
