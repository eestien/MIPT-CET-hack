from flask import Blueprint, render_template, jsonify, request



bp = Blueprint(__name__, __name__,template_folder='Templates')

def pr(data:str):
    return data

@bp.route('/')
def show():
    return render_template('index.html')










