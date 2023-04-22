from app.config.settings import Config
from app.create_app import create_app
from app.utils.ZMQBridge import entrypoint
from flask_sockets import Sockets
from gevent import pywsgi
# from geventwebsocket.handler import WebSocketHandler
from app.utils.ZMQBridge import WebSocketHandler

app = create_app(Config)

if __name__ == "__main__":
    entrypoint()
    # app.run(host='0.0.0.0')
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    print('server start')
    server.serve_forever()
    # app.serve_forever()

