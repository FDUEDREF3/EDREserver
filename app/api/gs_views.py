import os
from pathlib import Path
from datetime import datetime
import shutil
import threading

from flask import Blueprint
from flask import request, jsonify
from werkzeug.datastructures import FileStorage

from app import db,app
from app.api.views import getAllPerImgs, writeImgs 
from app.models.projectList import ProjectList

gs_api=Blueprint('gs',__name__)

imgType_list = {'jpg','bmp','png','jpeg','rgb','tif'}


@gs_api.route('/createProject', methods=["POST"])
def createProject():
    trainData = request.form
    projectName = trainData.get("title")
    avatar = request.files.get("avatar")
    dateTimeString = trainData.get("datetime")
    formatString = '%Y-%m-%d' 
    dateTimeObj = datetime.strptime(dateTimeString, formatString)

    state = 0   #0，1，2分别代表colmap中，训练中，训练结束
    avatarPathHead = "./app/gs_data/"+projectName+"/avatar/"
    imagePathHead = "./app/gs_data/"+projectName+"/images/"
    colmapDir = "./app/gs_data/"+projectName
    if not os.path.exists(colmapDir):
        os.mkdir(colmapDir)
    if not os.path.exists(avatarPathHead):
        os.mkdir(avatarPathHead)
    if not os.path.exists(imagePathHead):
        os.mkdir(imagePathHead)
        
    avatarPath = avatarPathHead +projectName+ Path(avatar.filename).suffix
    avatar.save(avatarPath)

    projectList = ProjectList(title=projectName, avatarImgPath=avatarPath, projectPath=imagePathHead, imgNum=0, createTime=dateTimeObj.date(), state=state, configPath=str(''), colmapPath=colmapDir,method=2)
    db.session.add(projectList)
    db.session.commit()

    return jsonify({'status': 'success'})


@gs_api.route('/uploadImgs', methods=["POST"])
def uploadImgs():
    title = request.form.get("title")
    images = request.files.getlist("imageFiles")

    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title #?
    imagePathHead = proj.projectPath
    if len(imagePathHead) == 0:
        return jsonify({'status': 'fail'})

    for imgs in images:
        imgs_name = os.path.basename(imgs.filename)
        imgs.save(imagePathHead  + '/' + imgs_name)

    return jsonify({'status': 'success'})


# def colmapthread(imagePath,type,projectName,pano):
#     if pano:
#         imgpathList=os.listdir(imagePath)
#         originPath = os.path.join(imagePath,'origin')
#         os.mkdir(originPath)
#         for i in imgpathList:
#             shutil.move(os.path.join(imagePath,i),originPath)
#         imgNewpathList=os.listdir(originPath)
#         for i in imgNewpathList:
#             imgpath1=os.path.join(originPath,i)
#             imgList=getAllPerImgs(imgpath1)
#             writeImgs(os.path.join(imagePath,i),imgList)
#         # 如果是全景图像则修改imagepath位置
#     with app.app_context():
#         proj = ProjectList.query.filter(ProjectList.title==projectName).first()
#         proj.state = 1
#         db.session.commit()
#     mainF(imagePath,type)
#     # print("colmap完成")
#     """colmap后修改数据库"""
#     with app.app_context():
#         proj = ProjectList.query.filter(ProjectList.title==projectName).first()
#         proj.state = 2
#         db.session.commit()

@gs_api.route('/testColmap',methods=["POST"])
def testColmap():
    title=request.form.get('title')
    print('1111111111111111111111111')
    cmdString="bash ./app/gs_data/colmap.sh " +title
    os.system(cmdString)
    return 'ok'
    


def colmapAndTrainThread(projectName,pano):
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        imagePath=proj.projectPath
    """colmap部分"""
    #TODO:还没对全景图路径做适配
    if pano:
        imgpathList=os.listdir(imagePath)
        for i in imgpathList:
            imgpath1=os.path.join(imagePath,i)
            imgList=getAllPerImgs(imgpath1)
            writeImgs(imgpath1,imgList)
        # 如果是全景图像则修改imagepath位置
        imagePath = os.path.join(imagePath,'TransPer')

    # ColmapCmdString="bash ./app/gs_data/colmap.sh " +projectName
    # os.system(ColmapCmdString)
    # print('colmap部分finish')
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()

    """训练部分"""
    """命令行运行"""
    #TODO:太拙劣了，到时换一下
    TrainCmdString='python ./app/gs_script/train.py -s ./app/gs_data/'+ projectName +'/dense -m' +'./app/gs_data/'+projectName+'/output1/'
    # For example : python ./app/gs_script/train.py -s ../gs_data/lvzi/dense -m ../gs_data/lvzi/output

    os.system(TrainCmdString)


    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        # config_path = Path(finalOutputPathHead + projectName + "/nerfacto/" + "config.yml")
        # proj.configPath = str(config_path)
        proj.state = 2
        db.session.commit()

@gs_api.route('/runColmapAndTrain', methods=["POST"])
def runColmapAndTrain():
    title = request.form.get("title")
    tpano = int(request.form.get("pano"))
    print(tpano)
    pano=(tpano>0)
    print(pano)
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    project_path = proj.projectPath

    if len(project_path) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=colmapAndTrainThread, args=(title,pano))
    thread.start()
    return jsonify({'status': 'success'})

@gs_api.route('/viewer', methods=["POST"])
def startViewer():
    title = request.form.get("title")
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    state = proj.state
    if not title:
        return jsonify({'status': 'fail'})
    if(state<2):
        return jsonify({'status': 'fail'})
    viewerPath='/home/dcy/code/EDREserver/app/gs_data/'+title+'/output'
    return jsonify({'data':viewerPath})
    