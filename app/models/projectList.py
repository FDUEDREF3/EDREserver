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
    state = db.Column(db.Integer)   # 0，1，2分别代表colmap中，训练中，训练结束
    configPath = db.Column(db.String)
    colmapPath = db.Column(db.Text)
    method = db.Column(db.Integer)  #0,1分别代表使用nerfstudio和nerf2mesh
    

    # repr()方法显示一个可读字符串
    def __repr__(self):
     return '<ProjectList {}>'.format(self.title)
