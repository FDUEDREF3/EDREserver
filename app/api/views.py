from flask import Blueprint  # 从flask包里面导入Flask核心类
from app import db
from app.models.projectList import ProjectList
from datetime import datetime



api = Blueprint('api', __name__)

@api.route('/h',methods=["GET"])  # 使用app提供的route装饰器 对函数进行装饰 即可成为视图函数
def hello():
    return 'hello flask'

@api.route('/selecttest', methods=["GET"])  # 测试从数据库中获取数据
def getData():
    # 添加数据
    projectList = ProjectList.query.filter(ProjectList.id=="1").first()
    print(projectList)
    return "数据操作成功"

@api.route('/addtest', methods=["POST"])  # 测试向数据库中添加数据
def addData():
    # 添加数据
    projectList = ProjectList(projectName="test2", previewImgPath="./data/proj1/test2", projectPath="./dataproj1", imgNum=200, createTime=datetime.now())
    db.session.add(projectList)
    db.session.commit()

    # # 查询数据
    # article = Article.query.filter_by(id=1)[0]
    # print(article.title)
    #
    # # 修改数据
    # article = Article.query.filter_by(id=1)[0]
    # article.content = "yyy"
    # db.session.commit()
    #
    # # 删除数据
    # article = Article.query.filter_by(id=1)[0]
    # db.session.delete(article)
    # db.session.commit()
    return "数据操作成功"