from flask import Blueprint  # 从flask包里面导入Flask核心类

api = Blueprint('api', __name__)

@api.route('/h')  # 使用app提供的route装饰器 对函数进行装饰 即可成为视图函数
def hello():
    return 'hello flask'