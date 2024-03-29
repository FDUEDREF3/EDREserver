from flask import Blueprint  # 从flask包里面导入Flask核心类
from flask import request, send_file,jsonify
from multiprocessing import Process

from app import db,app
from app.models.projectList import ProjectList
from datetime import datetime
from pathlib import Path
import os
import cv2

import multiprocessing
from werkzeug.datastructures import FileStorage
import base64
import concurrent.futures
import threading
import dill
from multiprocessing import context
import subprocess
import app.scripts.Equirec2Perspec as E2P 

from app.nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset 
from app.nerfstudio.process_data.images_to_nerstudio_dataset import ImagesToNerfstudioDataset
from app.scripts.train import main as starTrainMethod

from app.nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from app.nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from app.nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from app.nerfstudio.models.nerfacto import NerfactoModelConfig
from app.nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from app.nerfstudio.engine.optimizers import AdamOptimizerConfig
from app.nerfstudio.configs.base_config import ViewerConfig
from app.scripts.viewer.run_viewer import RunViewer
from concurrent.futures import ThreadPoolExecutor
import time

from app.nerfstudio.engine.trainer import TrainerConfig


api = Blueprint('api', __name__)


"""临时用"""
processDict = {}
availPort = {'7007':'','7008':''}


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



def getAllPerImgs(imgpath):
    equ = E2P.Equirectangular(imgpath)
    imgList=[]
    for j in range(0,6):
        img = equ.GetPerspective(80,j*60,0, 1080, 1920) # Specify parameters(FOV, theta, phi, height, width)
        imgList.append(img)    
    # img = equ.GetPerspective(60,0, 90, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
    # imgList.append(img)
    # img = equ.GetPerspective(60,0, 270, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
    # imgList.append(img)
    return imgList
def writeImgs(imgpath,imgList):
    pathdic=os.path.split(imgpath)
    if os.path.exists(os.path.join(pathdic[0],'TransPer')) is not True:
        os.mkdir(os.path.join(pathdic[0],'TransPer'))
    for img,idx in zip(imgList,range(0,len(imgList))):
        cv2.imwrite(os.path.join(os.path.join(pathdic[0],'TransPer'), pathdic[1][:pathdic[1].find('.')]+'_'+str(idx))+pathdic[1][pathdic[1].find('.'):],img)




@api.route('/getAllProjects', methods=["get"])   #获取所有项目
def getAllProjects():
    try:
        projectList = ProjectList.query.all()
    except:
        return jsonify({'status': 'fail', 'message':'database error'})
    reProjectList = []
    for proj in projectList:
        with open(proj.avatarImgPath, 'rb') as file:
            avatarImg = base64.b64encode(file.read()).decode('utf-8')
            projDict = {"id": proj.id, "title": proj.title, "avatar": avatarImg, "datetime": proj.createTime, "state": proj.state, "imgNum": proj.imgNum, "method": proj.method}
            reProjectList.append(projDict)
    return jsonify({'projects': reProjectList})


def trainthread(imagePathHead, outputPathHead, finalOutputPathHead, projectName):
    imagesToNerfstudioDataset = ImagesToNerfstudioDataset(Path(imagePathHead + projectName), Path(outputPathHead + projectName))
    # imagesToNerfstudioDataset.aquireData(Path(imagePathHead + projectName), Path(outputPathHead)) #增加数据，目前不需要
    imagesToNerfstudioDataset.main()
    # print("colmap完成")
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()

    """调包运行"""
    dataParser = NerfstudioDataParserConfig()
    dataParser.getDataDir(Path(outputPathHead + projectName))

    nowMethod = TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=dataParser,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        # vis="viewer",
        vis="tensorboard",           #不进行前端显示
    )
    nowMethod.set_output_dir(Path(finalOutputPathHead + projectName))
    starTrainMethod(nowMethod)


    """命令行运行"""
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " " + "--viewer.quit-on-train-completion True"
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " --vis tensorboard " + "--viewer.quit-on-train-completion True"
    # os.system(commandString)
    # p = subprocess.Popen(['python', '/home/dcy/code/EDREserver/app/scripts/train.py nerfacto','--data', outputPathHead + projectName, "--output-dir", finalOutputPathHead + projectName, "--viewer.quit-on-train-completion True"])
    
    # print("训练完成")
    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        config_path = Path(finalOutputPathHead + projectName + "/nerfacto/" + "config.yml")
        proj.configPath = str(config_path)
        proj.state = 2
        db.session.commit()
    # return "训练完成"

# def callback(result):
#     print('Result:', result)


@api.route('/startTrain', methods=["POST"])  # 测试向数据库中添加数据
def startTrain():
    try:
        trainData = request.form
        projectName = trainData.get("title")
        images = request.files.getlist("imageFiles")
        avatar = request.files.get("avatar")
        dateTimeString = trainData.get("datetime")
        formatString = '%Y-%m-%d' 
        dateTimeObj = datetime.strptime(dateTimeString, formatString)
    except:
        return jsonify({'status': 'fail', 'message':'parameters error'})
    

    state = 0   #0，1，2分别代表colmap中，训练中，训练结束

    imagePathHead = "./app/data/pureImages/"
    outputPathHead = "./app/data/afterColmap/"
    finalOutputPathHead = "./app/data/afterNerfacto/"
    avatarPathHead = "./app/data/avatar/"
    if not os.path.exists(imagePathHead):
        os.mkdir(imagePathHead)
    if not os.path.exists(outputPathHead):
        os.mkdir(outputPathHead)
    if not os.path.exists(finalOutputPathHead):
        os.mkdir(finalOutputPathHead)
    if not os.path.exists(avatarPathHead):
        os.mkdir(avatarPathHead)

    if not os.path.exists(imagePathHead + projectName + '/'):
        os.mkdir(imagePathHead + projectName + '/')
    for imgs in images:
        imgs_name = os.path.basename(imgs.filename)
        imgs.save(imagePathHead + projectName + '/' + imgs_name)

    avatarPath = avatarPathHead + projectName + Path(avatar.filename).suffix
    avatar.save(avatarPath)
    
    try:
        projectList = ProjectList(title=projectName, avatarImgPath=avatarPath, projectPath=imagePathHead+projectName, imgNum=len(images), createTime=dateTimeObj.date(), state=state, configPath=str(''), colmapPath=outputPathHead+projectName)
        db.session.add(projectList)
        db.session.commit()
    except:
        return jsonify({'status': 'fail', 'message':'database error'})

    # process = Process(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
    # process.start()
    thread = threading.Thread(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
    thread.start()

    return jsonify({'status': 'success'})

    # future = executor.submit(trainthread, imagePathHead, outputPathHead, finalOutputPathHead, projectName)
    # thread = threading.Thread(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
    # thread.start()
    # parent_conn, child_conn = multiprocessing.Pipe()
    # ctx = multiprocessing.get_context()
    # with ctx.Process(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName)) as proc:
    #     proc.start()
    #     result = parent_conn.recv()
    #     callback(result)
    #     proc.join()
    # future = executor.submit(trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
    # result = future.result()
    # print(result)
    # args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName)
    # trainthread(imagePathHead, outputPathHead, finalOutputPathHead, projectName)
    # executor.submit(lambda p: trainthread(*p), args)

    # projectList = ProjectList(projectName="test2", previewImgPath="./data/proj1/test2", projectPath="./dataproj1", imgNum=200, createTime=datetime.now())
    # db.session.add(projectList)
    # db.session.commit()


# def viewerthread(config_path):
#     """命令行运行"""
#     #commandString = "python /home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py " + "--load-config " + config_path
#     # os.system(commandString)


#     """调包运行"""
#     # runViewer = RunViewer(Path(config_path))
#     # runViewer.main()


@api.route('/viewer', methods=["POST"])  # 测试向数据库中添加数据
def startViewer():
    # title = request.args["title"]

    try:
        title = request.form.get("title")
    except:
        return jsonify({'status': 'fail', 'message':'parameters error', 'websocket_url':''})
    try:
        proj = ProjectList.query.filter(ProjectList.title==title).first()
        config_path = proj.configPath
    except:
        return jsonify({'status': 'fail', 'message':'database error', 'websocket_url':''})
    if len(config_path) == 0:
        return jsonify({'status': 'fail', 'message':'empty data', 'websocket_url':''})
    address = "10.177.35.76"
    port = ''
    """限制端口"""

    if title in processDict:
        return jsonify({'status': 'fail', 'message':'websocket already in use', 'websocket_url':''})
    if len(processDict)>=2:
        return jsonify({'status': 'fail', 'message':'full process', 'websocket_url':''})
    if availPort['7007'] == '':
        port = '7007'
    else:
        if availPort['7008'] == '':
            port = '7008'
    availPort[port] = title

    """异步调用"""
    # process = Process(target=viewerthread, args=(config_path,))
    # process.start()
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py " + "--load-config " + config_path
    # p = subprocess.Popen(['python', '/home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py','--load-config', config_path], 
    #                      stdout = subprocess.PIPE,
    #                      stderr = subprocess.PIPE,
    #                      universal_newlines=True,
    #                      shell=True)
    # print(config_path)
    nowpath = os.getcwd()
    viewer_dir = os.path.join(nowpath,'app/scripts/viewer/run_viewer.py')
    p = subprocess.Popen(['python', viewer_dir,'--load-config', config_path,'--viewer.websocket-port',port])


    """限制端口"""
    processDict[title] = p
    # time.sleep(5)
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py " + "--load-config " + config_path
    # os.system(commandString)
    # runViewer = RunViewer(Path(config_path))
    # runViewer.main()
    # p.kill()    
    # print (p.stdout.read())    

    websocket_url = "ws://" + address + ":" + port

    return jsonify({'status': 'success', 'message':'websocket is already open', 'websocket_url': websocket_url})



@api.route('/viewerClose', methods=["POST"])  
def stopViewer():
    # title = request.args["title"]
    try:
        title = request.form.get("title")
    except: 
        return jsonify({'status': 'fail', 'message':'parameters error'})
    try:
        p = processDict.pop(title)
        p.kill()
        for po in availPort.keys():
            if(availPort[po] == title):
                availPort[po] = ''
                
    except:
        return jsonify({'status': 'fail', 'message':'can not kill process: ' + str(title)})
    return jsonify({'status': 'success'})



@api.route('/createProject', methods=["POST"])
def createProject():
    trainData = request.form
    projectName = trainData.get("title")
    avatar = request.files.get("avatar")
    dateTimeString = trainData.get("datetime")
    formatString = '%Y-%m-%d' 
    dateTimeObj = datetime.strptime(dateTimeString, formatString)

    state = 0   #0，1，2分别代表colmap中，训练中，训练结束
    avatarPathHead = "./app/data/avatar/"
    imagePathHead = "./app/data/pureImages/"
    outputPathHead = "./app/data/afterColmap/"
    if not os.path.exists("./app/data"):
        os.mkdir("./app/data")
    if not os.path.exists(avatarPathHead):
        os.mkdir(avatarPathHead)
    if not os.path.exists(imagePathHead):
        os.mkdir(imagePathHead)
    if not os.path.exists(outputPathHead):
        os.mkdir(outputPathHead)
    if not os.path.exists("./app/data/afterNerfacto/"):
        os.mkdir("./app/data/afterNerfacto/")
    if not os.path.exists(imagePathHead + projectName + '/'):
        os.mkdir(imagePathHead + projectName + '/')
    avatarPath = avatarPathHead + projectName + Path(avatar.filename).suffix
    avatar.save(avatarPath)

    projectList = ProjectList(title=projectName, avatarImgPath=avatarPath, projectPath=imagePathHead+projectName, imgNum=0, createTime=dateTimeObj.date(), state=state, configPath=str(''), colmapPath=outputPathHead+projectName, method=0)
    db.session.add(projectList)
    db.session.commit()

    return jsonify({'status': 'success'})

@api.route('/uploadImgs', methods=["POST"])
def uploadImgs():
    title = request.form.get("title")
    images = request.files.getlist("imageFiles")

    imagePathHead = "./app/data/pureImages/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    project_path = proj.projectPath
    if len(project_path) == 0:
        return jsonify({'status': 'fail'})

    for imgs in images:
        imgs_name = os.path.basename(imgs.filename)
        imgs.save(imagePathHead + title + '/' + imgs_name)

    return jsonify({'status': 'success'})

def colmapthread(imagePath, outputPath, projectName,pano):
    
    if pano:
        imgpathList=os.listdir(imagePath)
        for i in imgpathList:
            imgpath1=os.path.join(imagePath,i)
            imgList=getAllPerImgs(imgpath1)
            writeImgs(imgpath1,imgList)
        # 如果是全景图像则修改imagepath位置
        imagePath = os.path.join(imagePath,'TransPer')

    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 0
        db.session.commit()
    imagesToNerfstudioDataset = ImagesToNerfstudioDataset(Path(imagePath), Path(outputPath))
    # imagesToNerfstudioDataset.aquireData(Path(imagePathHead + projectName), Path(outputPathHead)) #增加数据，目前不需要
    imagesToNerfstudioDataset.main()
    # print("colmap完成")
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()

def nerfactothread(colmapPath,finalOutputPathHead,projectName):
    """调包运行"""
    dataParser = NerfstudioDataParserConfig()
    dataParser.getDataDir(Path(colmapPath))

    nowMethod = TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=dataParser,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        # vis="viewer",
        vis="tensorboard",           #不进行前端显示
    )
    nowMethod.set_output_dir(Path(finalOutputPathHead + projectName))
    starTrainMethod(nowMethod)


    """命令行运行"""
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " " + "--viewer.quit-on-train-completion True"
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " --vis tensorboard " + "--viewer.quit-on-train-completion True"
    # os.system(commandString)
    # p = subprocess.Popen(['python', '/home/dcy/code/EDREserver/app/scripts/train.py nerfacto','--data', outputPathHead + projectName, "--output-dir", finalOutputPathHead + projectName, "--viewer.quit-on-train-completion True"])
    
    # print("训练完成")
    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        config_path = Path(finalOutputPathHead + projectName + "/nerfacto/" + "config.yml")
        proj.configPath = str(config_path)
        proj.state = 2
        db.session.commit()


def colmapAndTrainThread(imagePath, colmapOutputPath, finalOutputPathHead, projectName,pano):
    """colmap部分"""
    if pano:
        imgpathList=os.listdir(imagePath)
        for i in imgpathList:
            imgpath1=os.path.join(imagePath,i)
            imgList=getAllPerImgs(imgpath1)
            writeImgs(imgpath1,imgList)
        # 如果是全景图像则修改imagepath位置
        imagePath = os.path.join(imagePath,'TransPer')
    imagesToNerfstudioDataset = ImagesToNerfstudioDataset(Path(imagePath), Path(colmapOutputPath))
    # imagesToNerfstudioDataset.aquireData(Path(imagePathHead + projectName), Path(outputPathHead)) #增加数据，目前不需要
    imagesToNerfstudioDataset.main()
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()

    """训练部分"""
    """调包运行"""
    dataParser = NerfstudioDataParserConfig()
    dataParser.getDataDir(Path(colmapOutputPath))

    nowMethod = TrainerConfig(
        method_name="nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=dataParser,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        # vis="viewer",
        vis="tensorboard",           #不进行前端显示
    )
    nowMethod.set_output_dir(Path(finalOutputPathHead + projectName))
    starTrainMethod(nowMethod)
    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        config_path = Path(finalOutputPathHead + projectName + "/nerfacto/" + "config.yml")
        proj.configPath = str(config_path)
        proj.state = 2
        db.session.commit()


@api.route('/runColmap', methods=["POST"])
def runColmap():
    title = request.form.get("title")
    tpano = int(request.form.get("pano"))
    pano = False
    if tpano>0:
        pano=True
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    project_path = proj.projectPath
    outputPath = proj.colmapPath

    if len(project_path) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=colmapthread, args=(project_path, outputPath, title,pano))
    thread.start()
    return jsonify({'status': 'success'})

@api.route('/runTrain', methods=["POST"])
def runTrain():
    title = request.form.get("title")
    finalOutputPathHead = "./app/data/afterNerfacto/"
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    colmapPath = proj.colmapPath

    if len(colmapPath) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=nerfactothread, args=(colmapPath, finalOutputPathHead, title))
    thread.start()
    return jsonify({'status': 'success'})
    
@api.route('/runColmapAndTrain', methods=["POST"])
def runColmapAndTrain():
    title = request.form.get("title")
    tpano = int(request.form.get("pano"))
    print(tpano)
    pano = False
    if tpano>0:
        pano=True
    print(pano)
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    project_path = proj.projectPath
    outputPath = proj.colmapPath
    finalOutputPathHead = "./app/data/afterNerfacto/"   #训练完成结果路径

    if len(project_path) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=colmapAndTrainThread, args=(project_path, outputPath, finalOutputPathHead, title,pano))
    thread.start()
    return jsonify({'status': 'success'})
