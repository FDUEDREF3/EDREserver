from flask import Flask  # 从flask包里面导入Flask核心类
from app.api.views import api

app = Flask(__name__)  # 实例化flask核心对象

def register_blueprint(app):
    app.register_blueprint(api, url_prefix='/api')

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    register_blueprint(app)
    return app


if __name__ == '__main__':  # 当python解释器直接运行此文件的时候，里面的代码会执行
    app.run()

