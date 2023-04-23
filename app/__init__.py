from flask_sqlalchemy import SQLAlchemy
from flask import Flask  # 从flask包里面导入Flask核心类

db = SQLAlchemy()
app = Flask(__name__)  # 实例化flask核心对象
