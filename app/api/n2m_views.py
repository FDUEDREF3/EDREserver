from flask import Blueprint  # 从flask包里面导入Flask核心类
from flask import request, send_file,jsonify
from multiprocessing import Process

from app import db,app
from app.models.projectList import ProjectList
from datetime import datetime
from pathlib import Path
import os
import shutil
import app.scripts.Equirec2Perspec as E2P 

import base64
import threading
import subprocess
import zipfile
import cv2
import imghdr

from app.n2m_script.data.colmap2nerf import mainF

n2m_api = Blueprint('nerf2mesh', __name__)

imgType_list = {'jpg','bmp','png','jpeg','rgb','tif'}


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


# @n2m_api.route('/startTrain', methods=["POST"])  # 测试向数据库中添加数据
# def startTrain():
#     try:
#         trainData = request.form
#         projectName = trainData.get("title")
#         images = request.files.getlist("imageFiles")
#         avatar = request.files.get("avatar")
#         dateTimeString = trainData.get("datetime")
#         formatString = '%Y-%m-%d' 
#         dateTimeObj = datetime.strptime(dateTimeString, formatString)
#     except:
#         return jsonify({'status': 'fail', 'message':'parameters error'})
    

#     state = 0   #0，1，2分别代表colmap中，训练中，训练结束

#     imagePathHead = "./app/data/pureImages/"
#     outputPathHead = "./app/data/afterColmap/"
#     finalOutputPathHead = "./app/data/afterNerfacto/"
#     avatarPathHead = "./app/data/avatar/"
#     if not os.path.exists(imagePathHead):
#         os.mkdir(imagePathHead)
#     if not os.path.exists(outputPathHead):
#         os.mkdir(outputPathHead)
#     if not os.path.exists(finalOutputPathHead):
#         os.mkdir(finalOutputPathHead)
#     if not os.path.exists(avatarPathHead):
#         os.mkdir(avatarPathHead)

#     if not os.path.exists(imagePathHead + projectName + '/'):
#         os.mkdir(imagePathHead + projectName + '/')
#     for imgs in images:
#         imgs_name = os.path.basename(imgs.filename)
#         imgs.save(imagePathHead + projectName + '/' + imgs_name)

#     avatarPath = avatarPathHead + projectName + Path(avatar.filename).suffix
#     avatar.save(avatarPath)
    
#     try:
#         projectList = ProjectList(title=projectName, avatarImgPath=avatarPath, projectPath=imagePathHead+projectName, imgNum=len(images), createTime=dateTimeObj.date(), state=state, configPath=str(''), colmapPath=outputPathHead+projectName)
#         db.session.add(projectList)
#         db.session.commit()
#     except:
#         return jsonify({'status': 'fail', 'message':'database error'})

#     # process = Process(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
#     # process.start()
#     thread = threading.Thread(target=trainthread, args=(imagePathHead, outputPathHead, finalOutputPathHead, projectName))
#     thread.start()

#     return jsonify({'status': 'success'})




# @n2m_api.route('/viewer', methods=["POST"])  # 测试向数据库中添加数据
# def startViewer():
#     # title = request.args["title"]

#     try:
#         title = request.form.get("title")
#     except:
#         return jsonify({'status': 'fail', 'message':'parameters error', 'websocket_url':''})
#     try:
#         proj = ProjectList.query.filter(ProjectList.title==title).first()
#         config_path = proj.configPath
#     except:
#         return jsonify({'status': 'fail', 'message':'database error', 'websocket_url':''})
#     if len(config_path) == 0:
#         return jsonify({'status': 'fail', 'message':'empty data', 'websocket_url':''})
#     address = "10.177.35.162"
#     port = ''
#     """限制端口"""

#     if title in processDict:
#         return jsonify({'status': 'fail', 'message':'websocket already in use', 'websocket_url':''})
#     if len(processDict)>=2:
#         return jsonify({'status': 'fail', 'message':'full process', 'websocket_url':''})
#     if availPort['7007'] == '':
#         port = '7007'
#     else:
#         if availPort['7008'] == '':
#             port = '7008'
#     availPort[port] = title

#     """异步调用"""
#     # process = Process(target=viewerthread, args=(config_path,))
#     # process.start()
#     # commandString = "python /home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py " + "--load-config " + config_path
#     # p = subprocess.Popen(['python', '/home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py','--load-config', config_path], 
#     #                      stdout = subprocess.PIPE,
#     #                      stderr = subprocess.PIPE,
#     #                      universal_newlines=True,
#     #                      shell=True)
#     # print(config_path)
#     nowpath = os.getcwd()
#     viewer_dir = os.path.join(nowpath,'app/scripts/viewer/run_viewer.py')
#     p = subprocess.Popen(['python', viewer_dir,'--load-config', config_path,'--viewer.websocket-port',port])


#     """限制端口"""
#     processDict[title] = p
#     # time.sleep(5)
#     # commandString = "python /home/dcy/code/EDREserver/app/scripts/viewer/run_viewer.py " + "--load-config " + config_path
#     # os.system(commandString)
#     # runViewer = RunViewer(Path(config_path))
#     # runViewer.main()
#     # p.kill()    
#     # print (p.stdout.read())    

#     websocket_url = "ws://" + address + ":" + port

#     return jsonify({'status': 'success', 'message':'websocket is already open', 'websocket_url': websocket_url})



@n2m_api.route('/viewerClose', methods=["POST"])  
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
    # if os.path.exists(os.path.join(pathdic[0],'TransPer')) is not True:
    #     os.mkdir(os.path.join(pathdic[0],'TransPer'))
    for img,idx in zip(imgList,range(0,len(imgList))):
        cv2.imwrite(os.path.join(pathdic[0], pathdic[1][:pathdic[1].find('.')]+'_'+str(idx))+pathdic[1][pathdic[1].find('.'):],img)



@n2m_api.route('/createProject', methods=["POST"])
def createProject():
    trainData = request.form
    projectName = trainData.get("title")
    avatar = request.files.get("avatar")
    dateTimeString = trainData.get("datetime")
    formatString = '%Y-%m-%d' 
    dateTimeObj = datetime.strptime(dateTimeString, formatString)

    state = 0   #0，1，2分别代表colmap中，训练中，训练结束
    avatarPathHead = "./app/n2m_data/"+projectName+"/avatar/"
    imagePathHead = "./app/n2m_data/"+projectName+"/images/"
    colmapDir = "./app/n2m_data/"+projectName
    if not os.path.exists(colmapDir):
        os.mkdir(colmapDir)
    if not os.path.exists(avatarPathHead):
        os.mkdir(avatarPathHead)
    if not os.path.exists(imagePathHead):
        os.mkdir(imagePathHead)
        
    avatarPath = avatarPathHead +projectName+ Path(avatar.filename).suffix
    avatar.save(avatarPath)

    projectList = ProjectList(title=projectName, avatarImgPath=avatarPath, projectPath=imagePathHead, imgNum=0, createTime=dateTimeObj.date(), state=state, configPath=str(''), colmapPath=colmapDir,method=1)
    db.session.add(projectList)
    db.session.commit()

    return jsonify({'status': 'success'})

@n2m_api.route('/uploadImgs', methods=["POST"])
def uploadImgs():
    title = request.form.get("title")
    images = request.files.getlist("imageFiles")

    
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    imagePathHead = proj.projectPath
    if len(imagePathHead) == 0:
        return jsonify({'status': 'fail'})

    for imgs in images:
        imgs_name = os.path.basename(imgs.filename)
        imgs.save(imagePathHead  + '/' + imgs_name)

    return jsonify({'status': 'success'})

def colmapAndTrainthread(imagePath,type,projectName,colmapPath,pano):
    if pano:
        imgpathList=os.listdir(imagePath)
        originPath = os.path.join(imagePath,'origin')
        os.mkdir(originPath)
        for i in imgpathList:
            shutil.move(os.path.join(imagePath,i),originPath)
        imgNewpathList=os.listdir(originPath)
        for i in imgNewpathList:
            imgpath1=os.path.join(originPath,i)
            imgList=getAllPerImgs(imgpath1)
            writeImgs(os.path.join(imagePath,i),imgList)
        # 如果是全景图像则修改imagepath位置
        # imagePath = os.path.join(imagePath,'TransPer')
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()
    mainF(imagePath,type)
    # print("colmap完成")
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 2
        db.session.commit()

    # commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 1 --dt_gamma 0 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --iters 10000 --decimate_target 1e5 --sdf"
    # os.system(commandString1)
    # commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 1 --dt_gamma 0 --stage 1 --iters 5000 --lambda_normal 1e-2 --refine_remesh_size 0.01 --sdf --ssaa 1 --texture_size 2048"
    # os.system(commandString2)
    try:
        commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --lambda_tv 2e-8 --visibility_mask_dilation 50"
        os.system(commandString1)
        commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --stage 1 --iters 10000 --lambda_normal 1e-3 --ssaa 1 --texture_size 2048"
        os.system(commandString2)
    except:
        os.removedirs(colmapPath+'/'+projectName)
        commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 4 --enable_cam_center --enable_cam_near_far --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --lambda_tv 2e-8 --visibility_mask_dilation 50"
        os.system(commandString1)
        commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 4 --enable_cam_center --enable_cam_near_far --stage 1 --iters 10000 --lambda_normal 1e-3 --ssaa 1 --texture_size 2048"
        os.system(commandString2)
    # print("训练完成")
    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        config_path = Path(colmapPath+'/'+projectName)
        proj.configPath = str(config_path)
        proj.state = 3
        db.session.commit()

def colmapthread(imagePath,type,projectName,pano):
    if pano:
        imgpathList=os.listdir(imagePath)
        originPath = os.path.join(imagePath,'origin')
        os.mkdir(originPath)
        for i in imgpathList:
            shutil.move(os.path.join(imagePath,i),originPath)
        imgNewpathList=os.listdir(originPath)
        for i in imgNewpathList:
            imgpath1=os.path.join(originPath,i)
            imgList=getAllPerImgs(imgpath1)
            writeImgs(os.path.join(imagePath,i),imgList)
        # 如果是全景图像则修改imagepath位置
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 1
        db.session.commit()
    mainF(imagePath,type)
    # print("colmap完成")
    """colmap后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        proj.state = 2
        db.session.commit()

def nerf2meshthread(colmapPath,finalOutputPathHead,projectName):
    """命令行运行"""
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " " + "--viewer.quit-on-train-completion True"
    # commandString = "python /home/dcy/code/EDREserver/app/scripts/train.py nerfacto " + "--data " + outputPathHead + projectName + " --output-dir " + finalOutputPathHead + projectName + " --vis tensorboard " + "--viewer.quit-on-train-completion True"
    # os.system(commandString)
    # p = subprocess.Popen(['python', '/home/dcy/code/EDREserver/app/scripts/train.py nerfacto','--data', outputPathHead + projectName, "--output-dir", finalOutputPathHead + projectName, "--viewer.quit-on-train-completion True"])
    # first impact
    # commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 1 --dt_gamma 0 --stage 0 --clean_min_f 16 --clean_min_d 10 --visibility_mask_dilation 50 --iters 10000 --decimate_target 1e5 --sdf"
    # os.system(commandString1)
    # commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 1 --dt_gamma 0 --stage 1 --iters 5000 --lambda_normal 1e-2 --refine_remesh_size 0.01 --sdf --ssaa 1 --texture_size 2048"
    # os.system(commandString2)
    try:
        commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --lambda_tv 2e-8 --visibility_mask_dilation 50"
        os.system(commandString1)
        commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 8 --enable_cam_center --enable_cam_near_far --stage 1 --iters 10000 --lambda_normal 1e-3 --ssaa 1 --texture_size 2048"
        os.system(commandString2)
    except:
        os.removedirs(colmapPath+'/'+projectName)
        commandString1 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 4 --enable_cam_center --enable_cam_near_far --stage 0 --lambda_entropy 1e-3 --clean_min_f 16 --clean_min_d 10 --lambda_tv 2e-8 --visibility_mask_dilation 50"
        os.system(commandString1)
        commandString2 = "python app/n2m_script/main.py "+colmapPath+" --workspace "+colmapPath+'/'+projectName+" -O --data_format colmap --bound 4 --enable_cam_center --enable_cam_near_far --stage 1 --iters 10000 --lambda_normal 1e-3 --ssaa 1 --texture_size 2048"
        os.system(commandString2)
    # print("训练完成")
    """训练完成后修改数据库"""
    with app.app_context():
        proj = ProjectList.query.filter(ProjectList.title==projectName).first()
        config_path = Path(colmapPath+'/'+projectName)
        proj.configPath = str(config_path)
        proj.state = 3
        db.session.commit()


@n2m_api.route('/runColmap', methods=["POST"])
def runColmap():
    # only colmap
    # title = request.form.get("title")
    # # outputPathHead = "./app/data/afterColmap/"
    # proj = ProjectList.query.filter(ProjectList.title==title).first()
    # title = proj.title
    # project_path = proj.projectPath

    # if len(project_path) == 0:
    #     return jsonify({'status': 'fail'})
    # thread = threading.Thread(target=colmapthread, args=(project_path,0, title))
    # thread.start()
    # return jsonify({'status': 'success'})

    title = request.form.get("title")
    tpano = int(request.form.get("pano"))
    pano = False
    if tpano>0:
        pano=True
    
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    project_path = proj.projectPath
    colmapPath = proj.colmapPath

    if len(project_path) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=colmapAndTrainthread, args=(project_path,0, title,colmapPath,pano))
    thread.start()
    return jsonify({'status': 'success'})

@n2m_api.route('/runTrain', methods=["POST"])
def runTrain():
    title = request.form.get("title")
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    colmapPath = proj.colmapPath

    if len(colmapPath) == 0:
        return jsonify({'status': 'fail'})
    thread = threading.Thread(target=nerf2meshthread, args=(colmapPath, colmapPath, title))
    thread.start()
    return jsonify({'status': 'success'})
    
# @n2m_api.route('/runColmapAndTrain', methods=["POST"])
# def runColmapAndTrain():
#     title = request.form.get("title")
#     # outputPathHead = "./app/data/afterColmap/"
#     proj = ProjectList.query.filter(ProjectList.title==title).first()
#     title = proj.title
#     project_path = proj.projectPath
#     outputPath = proj.colmapPath
#     finalOutputPathHead = "./app/data/afterNerfacto/"   #训练完成结果路径

#     if len(project_path) == 0:
#         return jsonify({'status': 'fail'})
#     thread = threading.Thread(target=colmapAndTrainThread, args=(project_path, outputPath, finalOutputPathHead, title))
#     thread.start()
#     return jsonify({'status': 'success'})

@n2m_api.route('/viewer', methods=["POST"])  # 发送mesh文件
def startViewer():
    viewer_path = "/home/dcy/code/React3DRE/public/mesh/"
    title = request.form.get("title")
    # outputPathHead = "./app/data/afterColmap/"
    proj = ProjectList.query.filter(ProjectList.title==title).first()
    title = proj.title
    colmapPath = proj.colmapPath
    state = proj.state
    if(state<3):
        return jsonify({'status': 'fail'})
    else:
        finalOut = proj.configPath
        stage1_path = finalOut+'/'+'mesh_stage1'
        # zip_name = title+'.zip'
        # with zipfile.ZipFile(zip_name, 'w') as zf:
        #     for file in os.listdir(stage1_path):
        #         file_path = os.path.join(stage1_path, file)
        #         if os.path.isfile(file_path):
        #             zf.write(file_path)
        #     zf = base64.b64encode(zf).decode('utf-8')
        #     re = {"title": proj.title, "mesh": zf}
        #     return jsonify({'results': re})
        if not os.path.exists(viewer_path+title):
            shutil.copytree(stage1_path, viewer_path+title)
        # current_path = os.getcwd()
        # s_path = os.path.join(current_path,stage1_path)
        s_path = os.path.join(viewer_path,title)
        return s_path
        
        
