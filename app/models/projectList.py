'''对应数据库记录项目列表'''
from app import db
from datetime import time

class ProjectList(db.Model):
    #表名
    __tablename__ = 'projectList'
    #定义列对象
    id = db.Column(db.Integer, primary_key=True)
    projectName = db.Column(db.Text)
    previewImgPath = db.Column(db.Text)
    projectPath = db.Column(db.Text)
    imgNum = db.Column(db.Integer)
    createTime = db.Column(db.DateTime)

    # repr()方法显示一个可读字符串
    def __repr__(self):
     return '<ProjectList {}>'.format(self.projectName)
