from flask import Blueprint, render_template, jsonify, request


bp = Blueprint(__name__, __name__,template_folder='Templates')

def pr(data:str):
    return data


@bp.route('/background_process')
def background_process():
    try:
        data = request.args.get("raw_data_url")
        if data:
            return jsonify(pr(data))
    except Exception:
        return "Enter data url, please"

