from flask import Flask  # 从flask包里面导入Flask核心类
from app.api.views import api
from app.config.settings import DBConfig
from flask_sqlalchemy import SQLAlchemy
from app import db

app = Flask(__name__)  # 实例化flask核心对象


def register_blueprint(app):
    app.register_blueprint(api, url_prefix='/api')

def connectDB(dbConfig):
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{dbConfig.USERNAME}:{dbConfig.PASSWORD}@{dbConfig.HOSTNAME}:{dbConfig.PORT}/{dbConfig.DATABASE}?charset=utf8mb4"

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    register_blueprint(app)
    # connectDB(DBConfig)
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DBConfig.USERNAME}:{DBConfig.PASSWORD}@{DBConfig.HOSTNAME}:{DBConfig.PORT}/{DBConfig.DATABASE}?charset=utf8mb4"
    db.init_app(app)
    return app


if __name__ == '__main__':  # 当python解释器直接运行此文件的时候，里面的代码会执行
    app.run()

