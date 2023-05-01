import sys
sys.path.append("./app")
from flask_sqlalchemy import SQLAlchemy
from flask import Flask  # 从flask包里面导入Flask核心类
from app.config.settings import Config
from app.create_app import create_app

from flask_sockets import Sockets
from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
from app.utils.ZMQBridge import WebSocketHandler
from app import app
if __name__ == "__main__":
    app = create_app(app, Config)
    # entrypoint()
    # print("test")
    # app.run(host='0.0.0.0')
    address = '0.0.0.0'
    web_port = 8080
    server = pywsgi.WSGIServer((address, web_port), app)
    # server = pywsgi.WSGIServer(('', 5010), app, handler_class=WebSocketHandler)
    web_url = str(address) + ':' + str(web_port)
    print('server start at: ' + web_url)
    server.serve_forever()
    # app.serve_forever()

