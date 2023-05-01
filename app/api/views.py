from flask import Blueprint  # 从flask包里面导入Flask核心类
from flask import request
from app import db
from app.models.projectList import ProjectList
from datetime import datetime
from pathlib import Path

from app.nerfstudio.process_data.video_to_nerfstudio_dataset import VideoToNerfstudioDataset 
from app.nerfstudio.process_data.images_to_nerstudio_dataset import ImagesToNerfstudioDataset
from app.scripts.train import main as starTrain
from app.nerfstudio.engine.trainer import TrainerConfig
from app.nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from app.nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from app.nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from app.nerfstudio.models.nerfacto import NerfactoModelConfig
from app.nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from app.nerfstudio.engine.optimizers import AdamOptimizerConfig
from app.nerfstudio.configs.base_config import ViewerConfig



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


@api.route('/startTrain', methods=["POST"])  # 测试向数据库中添加数据
def startTrain():
    trainData = request.form
    # projectName = trainData.get("name")
    # images = trainData.get("files")
    projectName = "poster"
    imagePathHead = "./app/data/pureImages/"
    outputPathHead = "./app/data/afterColmap/"
    finalOutputPathHead = "./app/data/afterNerfacto/"
    imagesToNerfstudioDataset = ImagesToNerfstudioDataset(Path(imagePathHead + projectName), Path(outputPathHead + projectName))
    # imagesToNerfstudioDataset.aquireData(Path(imagePathHead + projectName), Path(outputPathHead))
    # imagesToNerfstudioDataset.main()
    print("colmap完成")

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
        # viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        # vis="viewer",
        vis="tensorboard",           #不进行前端显示
    )
    nowMethod.set_output_dir(Path(finalOutputPathHead + projectName))
    starTrain(nowMethod)
    return "训练完成"


    # projectList = ProjectList(projectName="test2", previewImgPath="./data/proj1/test2", projectPath="./dataproj1", imgNum=200, createTime=datetime.now())
    # db.session.add(projectList)
    # db.session.commit()
