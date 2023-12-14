from flask import Flask  # 从flask包里面导入Flask核心类
from app.api.views import api
from app.api.n2m_views import n2m_api
from app.api.gs_views import gs_api
from app.config.settings import DBConfig
from flask_sqlalchemy import SQLAlchemy
from app import db
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
# from app.utils.ZMQBridge import ZMQWebSocketBridge
# sockets = Sockets(app)
def register_blueprint(app):
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(n2m_api, url_prefix='/nerf2mesh')
    app.register_blueprint(gs_api, url_prefix='/gs')
    

def connectDB(app, dbConfig):
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{dbConfig.USERNAME}:{dbConfig.PASSWORD}@{dbConfig.HOSTNAME}:{dbConfig.PORT}/{dbConfig.DATABASE}?charset=utf8mb4"

def create_app(app, config):
    # app.config['DEBUG'] = True
    app.config.from_object(config)
    register_blueprint(app)
    # connectDB(DBConfig)
    app.config[
        'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DBConfig.USERNAME}:{DBConfig.PASSWORD}@{DBConfig.HOSTNAME}:{DBConfig.PORT}/{DBConfig.DATABASE}?charset=utf8mb4"
    db.init_app(app)
    # server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    # app.config['bridge'] = ZMQWebSocketBridge
    return app
    # return server


# if __name__ == '__main__':  # 当python解释器直接运行此文件的时候，里面的代码会执行
    # app.run()
    # server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)



