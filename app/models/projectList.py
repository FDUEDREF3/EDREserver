'''对应数据库记录项目列表'''
from app import db
from datetime import time

class ProjectList(db.Model):
    #表名
    __tablename__ = 'projectList'
    #定义列对象
    id = db.Column(db.Integer, primary_key=True)
    # projectName = db.Column(db.Text)
    title = db.Column(db.String)
    avatarImgPath = db.Column(db.Text)
    projectPath = db.Column(db.Text)
    imgNum = db.Column(db.Integer)
    # createTime = db.Column(db.DateTime)  #带有时分秒
    createTime = db.Column(db.Date)
    state = db.Column(db.Integer)
<<<<<<< HEAD
    configPath = db.Column(db.String)

    # repr()方法显示一个可读字符串
    def __repr__(self):
     return '<ProjectList {}>'.format(self.title)

=======

    # repr()方法显示一个可读字符串
    def __repr__(self):
     return '<ProjectList {}>'.format(self.projectName)
>>>>>>> 8ac7fbad4e168e7df784aa83718d713ca4f15365
