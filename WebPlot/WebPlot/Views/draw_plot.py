from flask import Blueprint, render_template


bp = Blueprint(__name__, __name__,template_folder='Templates')


@bp.route('/plot')
def show():
    return render_template('draw_plot.html')