from flask import Flask
from WebPlot.Views.index import bp as index_bp
from WebPlot.Views.draw_plot import bp as draw_plot_bp
from WebPlot.Views.background_process import bp as background_process

app = Flask(__name__)

app.register_blueprint(index_bp)
app.register_blueprint(draw_plot_bp)
app.register_blueprint(background_process)