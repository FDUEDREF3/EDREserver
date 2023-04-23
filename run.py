from flask_sqlalchemy import SQLAlchemy
from flask import Flask  # 从flask包里面导入Flask核心类
from app.config.settings import Config
from app.create_app import create_app
from app.utils.ZMQBridge import entrypoint
from flask_sockets import Sockets
from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
from app.utils.ZMQBridge import WebSocketHandler
from app import app
if __name__ == "__main__":
    app = create_app(app, Config)
    entrypoint()
    # print("test")
    # app.run(host='0.0.0.0')
    # server = pywsgi.WSGIServer(('', 5010), app, handler_class=WebSocketHandler)
    # print('server start')
    # server.serve_forever()
    # app.serve_forever()

